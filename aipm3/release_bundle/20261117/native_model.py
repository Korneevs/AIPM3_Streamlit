"""Three separately supervised heads, one input per meaning, unchanged logistic link."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.svm import SVR

def numeric(s):
    if pd.api.types.is_numeric_dtype(s):return pd.to_numeric(s,errors='coerce')
    return pd.to_numeric(s.map(lambda v:1. if str(v).lower()=='true' else 0. if str(v).lower()=='false' else v),errors='coerce')

def reference_feature(train,query,mode):
    geometric='geo' in mode;hierarchical='hier' in mode
    smoothing=float(mode.split('hs')[1].split('_')[0]) if 'hs' in mode else 3.
    values=train.assign(y=np.log(np.maximum(train.y,.001))) if geometric else train
    family=values.groupby('family').y.mean().to_dict();brand={str(k):set(g.family) for k,g in train.groupby('brand',dropna=False)}
    vertical={str(k):set(g.family) for k,g in train.groupby('vertical',dropna=False)}
    total=sum(family.values());size=len(family);out=[]
    for f,k,v in zip(query.family,query.brand.astype(str),query.vertical.astype(str)):
        base=(total-family.get(f,0))/(size-int(f in family))
        if hierarchical:
            vg=vertical.get(v,set())-{f};base=(sum(family[g] for g in vg)+3*base)/(len(vg)+3)
        bg=brand.get(k,set())-{f}
        value=(sum(family[g] for g in bg)+smoothing*base)/(len(bg)+smoothing) if len(bg)+smoothing else base
        out.append(value)
    return np.exp(out) if geometric else np.array(out)

class Head:
    def __init__(self,state,estimator):self.state=state;self.estimator=estimator

    def raw(self,frame):
        s=self.state;mode=s['spec']['mode'];columns=s['columns'];parts=[]
        for c in columns:
            if c=='brand_history':v=reference_feature(pd.DataFrame(s['history']),frame,mode)
            else:
                source=c if c in frame else 'a2extra__'+c
                if source not in frame:raise ValueError('Missing feature: '+c)
                v=numeric(frame[source]).to_numpy(float)
                relative=(mode.endswith('_share') and c.endswith('_seconds')) or (mode.endswith('_firstshare') and c=='brand_first_mention_seconds')
                if relative:
                    dur='total_video_duration_sec' if 'total_video_duration_sec' in frame else 'a2extra__total_video_duration_sec'
                    v=v/np.maximum(1,numeric(frame[dur]).to_numpy(float))
            parts.append(v)
        return np.column_stack(parts)

    def design(self,frame):
        s=self.state;x=self.raw(frame);x=np.where(np.isfinite(x),x,s['median']);z=(x-s['center'])/s['scale'];step=s['spec'].get('step',0)
        if step=='recall_stable':
            for j,c in enumerate(s['columns']):
                if c in ['human_characters_count','silence_or_music_only_seconds']:z[:,j]=np.round(z[:,j])
        elif step:z=np.round(z/step)*step
        return z

    @classmethod
    def fit(cls,train,spec):
        assert len(spec['fixed'])==len(set(spec['fixed']))
        state=dict(spec=spec,columns=spec['fixed'],history=train[['family','brand','vertical','y']].to_dict('records'),training_families=sorted(train.family.unique()))
        result=cls(state,None);x=result.raw(train)
        median=np.array([np.nanmedian(v) if np.isfinite(v).any() else 0. for v in x.T]);x=np.where(np.isfinite(x),x,median)
        center=x.mean(0);scale=x.std(0);scale[scale<1e-9]=1.
        state.update(median=median.tolist(),center=center.tolist(),scale=scale.tolist())
        z=result.design(train);y0=train.y.to_numpy();target=spec['target']
        y=np.log(np.maximum(y0,.001)) if target=='log' else rankdata(y0)/(len(y0)+1) if target=='rank' else y0.copy()
        if spec.get('brand_offset'):
            j=state['columns'].index('brand_history');y-=np.log(np.maximum(.001,x[:,j]));z=np.delete(z,j,axis=1)
        mu=y.mean();std=max(y.std(),1e-8);y=(y-mu)/std
        w=1/train.groupby('family').family.transform('size').to_numpy()
        if 'avito_weight' in spec:
            total=w.sum();w*=np.where(train.brand.eq('Avito'),spec['avito_weight'],1.);w*=total/w.sum()
        if spec['learner']=='ridge':
            a=np.column_stack([np.ones(len(z)),z]);pen=np.eye(a.shape[1])*spec['alpha'];pen[0,0]=0
            estimator=np.linalg.solve(a.T@(w[:,None]*a)+pen,a.T@(w*y))
        else:
            assert spec['learner']=='svr'
            estimator=SVR(C=spec['C'],gamma=spec['gamma'],epsilon=spec.get('epsilon',.05));estimator.fit(z,y,sample_weight=w)
        state.update(target_mean=float(mu),target_scale=float(std),raw_y=y0.tolist())
        result.estimator=estimator;return result

    def predict(self,frame):
        s=self.state;sp=s['spec'];z=self.design(frame);offset=0.
        if sp.get('brand_offset'):
            j=s['columns'].index('brand_history');offset=np.log(np.maximum(.001,self.raw(frame)[:,j]));z=np.delete(z,j,axis=1)
        p=np.column_stack([np.ones(len(z)),z])@self.estimator if sp['learner']=='ridge' else self.estimator.predict(z)
        p=p*s['target_scale']+s['target_mean']+offset
        if sp['target']=='log':p=np.exp(np.clip(p,-12,3))
        elif sp['target']=='rank':p=np.quantile(s['raw_y'],np.clip(p,0,1))
        return np.clip(p,.001,1 if sp['task']!='r' else max(1,max(s['raw_y'])*1.2))

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        (path/'state.json').write_text(json.dumps(self.state,ensure_ascii=False,indent=2)+'\n')
        (path/'features.json').write_text(json.dumps(self.state['columns'],ensure_ascii=False,indent=2)+'\n')
        joblib.dump(self.estimator,path/'estimator.joblib',compress=3)

    @classmethod
    def load(cls,path):
        path=Path(path);return cls(json.loads((path/'state.json').read_text()),joblib.load(path/'estimator.joblib'))

def coefficient(q,reference_mean):
    assert np.isfinite(reference_mean) and reference_mean>0
    q=np.asarray(q,dtype=float);assert np.isfinite(q).all() and (q>=0).all()
    return 2/(1+np.exp(-(q/reference_mean-1)))

class AIPM3:
    def __init__(self,path=None):
        path=Path(path) if path else Path(__file__).resolve().parent/'models'
        self.heads={t:Head.load(path/t) for t in ['n','m','r']}
        cs=[set(h.state['columns']) for h in self.heads.values()]
        assert not (cs[0]&cs[1] or cs[0]&cs[2] or cs[1]&cs[2])
        assert sum(map(len,cs))>=15

    def score(self,noticeability,message_delivery,recall,reference_mean=None):
        frames={t:d.sort_values(['record','repeat']).reset_index(drop=True) for t,d in zip(['n','m','r'],[noticeability,message_delivery,recall])}
        keys=frames['n'][['record','repeat']]
        assert not keys.duplicated().any()
        assert all(keys.equals(d[['record','repeat']]) for d in frames.values())
        assert all(list(g.repeat)==list(range(1,11)) for _,g in keys.groupby('record'))
        p={t:self.heads[t].predict(d) for t,d in frames.items()}
        result=keys.assign(noticeability=p['n'],message_delivery=p['m'],norm_ad_recall=p['r'],OPM=p['n']*p['m'],Q=p['n']*p['m']*p['r'])
        out=result.groupby('record').mean(numeric_only=True).drop(columns='repeat').reset_index()
        if reference_mean is not None:out['C']=coefficient(out.Q,reference_mean)
        return out

"""Observable feature definitions; no targets or outcome-derived inputs."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
OriginalHead = Head


def relative(numerator, denominator):
    a=np.asarray(numerator,float);b=np.asarray(denominator,float)
    return np.divide(a,b,out=np.full_like(a,np.nan),where=np.abs(b)>1e-9)


class Head(OriginalHead):
    def predict(self,frame):
        if self.state['spec']['target']!='rank_records':return super().predict(frame)
        z=self.design(frame);st=self.state
        p=np.column_stack([np.ones(len(z)),z])@self.estimator if st['spec']['learner']=='ridge' else self.estimator.predict(z)
        p=p*st['target_scale']+st['target_mean']
        return np.clip(np.quantile(st['raw_y'],np.clip(p,0,1)),.001,1.)

    def design(self,frame):
        state=self.state
        if 'clip_lo' not in state:return super().design(frame)
        x=self.raw(frame);x=np.where(np.isfinite(x),x,state['median'])
        lo=np.asarray(state['clip_lo'],float);hi=np.asarray(state['clip_hi'],float)
        lo=np.where(np.isfinite(lo),lo,-np.inf);hi=np.where(np.isfinite(hi),hi,np.inf)
        x=np.clip(x,lo,hi)
        z=(x-state['center'])/state['scale'];step=state['spec'].get('step',0)
        if step:z=np.round(z/step)*step
        return z

    @classmethod
    def fit(cls,train,spec):
        if spec.get('clip_quantile',0) or spec.get('scaler')=='robust' or spec['target']=='rank_records':
            assert not spec.get('brand_offset')
            state=dict(spec=spec,columns=spec['fixed'],history=train[['family','brand','vertical','y']].to_dict('records'),training_families=sorted(train.family.unique()))
            result=cls(state,None);x=result.raw(train)
            median=np.array([np.nanmedian(z) if np.isfinite(z).any() else 0. for z in x.T]);x=np.where(np.isfinite(x),x,median)
            q=spec.get('clip_quantile',0);lo=np.quantile(x,q,axis=0);hi=np.quantile(x,1-q,axis=0)
            # q=0 means no clipping, including for queries outside training range.
            if q==0:lo[:]=-np.inf;hi[:]=np.inf
            x=np.clip(x,lo,hi);center=x.mean(0);std=x.std(0);std[std<1e-9]=1.
            scale=(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0))/1.349 if spec.get('scaler')=='robust' else std
            scale=np.where(scale<1e-9,std,scale)
            state.update(median=median.tolist(),center=center.tolist(),scale=scale.tolist(),
                clip_lo=[float(v) if np.isfinite(v) else None for v in lo],
                clip_hi=[float(v) if np.isfinite(v) else None for v in hi])
            z=result.design(train);raw=train.y.to_numpy();target=spec['target']
            y=np.log(np.maximum(raw,.001)) if target=='log' else rankdata(raw)/(len(raw)+1) if target=='rank' else raw.copy()
            calibration_y=raw
            if target=='rank_records':
                _,first,inverse=np.unique(train.record.to_numpy(),return_index=True,return_inverse=True)
                calibration_y=raw[first];y=(rankdata(calibration_y)/(len(calibration_y)+1))[inverse]
            mu=y.mean();std_y=max(y.std(),1e-8);y=(y-mu)/std_y
            w=1/train.groupby('family').family.transform('size').to_numpy()
            if spec['learner']=='ridge':
                a=np.column_stack([np.ones(len(z)),z]);pen=np.eye(a.shape[1])*spec['alpha'];pen[0,0]=0
                estimator=np.linalg.solve(a.T@(w[:,None]*a)+pen,a.T@(w*y))
            else:
                from sklearn.svm import SVR
                estimator=SVR(C=spec['C'],gamma=spec['gamma'],epsilon=spec.get('epsilon',.05))
                estimator.fit(z,y,sample_weight=w)
            state.update(target_mean=float(mu),target_scale=float(std_y),raw_y=calibration_y.tolist());result.estimator=estimator
            return result
        if spec['learner'] not in ['extra','cat']:
            return super().fit(train,spec)
        # Reuse the same fold-only preprocessing and target scaling.
        result=super().fit(train,dict(spec,learner='ridge',alpha=100))
        result.state['spec']=spec
        x=result.design(train);raw=train.y.to_numpy()
        y=np.log(np.maximum(raw,.001)) if spec['target']=='log' else rankdata(raw)/(len(raw)+1) if spec['target']=='rank' else raw
        y=(y-result.state['target_mean'])/result.state['target_scale']
        w=1/train.groupby('family').family.transform('size').to_numpy()
        estimator=tree_estimator(spec);estimator.fit(x,y,sample_weight=w)
        result.estimator=estimator
        return result

    def raw(self,frame):
        spec=self.state['spec'];cols=self.state['columns'];data=frame
        derived={'derived__opening_contrast_ratio':('phys__first3_contrast','phys__contrast_mean'),
            'derived__opening_motion_ratio':('phys__first3_motion_mean','phys__motion_mean')}
        if set(cols)&set(derived):
            data=frame.copy()
            for col,(a,b) in derived.items():
                if col in cols:data[col]=relative(numeric(frame[a]),numeric(frame[b]))
        if 'vertical_attention_history' in cols:
            data=data.copy();history=pd.DataFrame(self.state['history'])
            means=history.groupby('family').y.mean().to_dict()
            verticals={str(k):set(g.family) for k,g in history.groupby('vertical',dropna=False)}
            total=sum(means.values());values=[]
            for family,vertical in zip(frame.family,frame.vertical.astype(str)):
                other=verticals.get(vertical,set())-{family}
                fallback=(total-means.get(family,0))/(len(means)-int(family in means))
                values.append((sum(means[f] for f in other)+3*fallback)/(len(other)+3))
            data['vertical_attention_history']=values
        x=super().raw(data)
        if spec.get('state_rubric')=='legacy' and 'state_transformation_present' in cols:
            x[:,cols.index('state_transformation_present')]=numeric(frame['state_transformation']).to_numpy(float)
        if spec.get('round_time_band') and 'panel__first_core_claim_time_band' in cols:
            j=cols.index('panel__first_core_claim_time_band');x[:,j]=np.floor(x[:,j]+.5)
        if spec.get('feature_definition')=='relative_visual':
            for c,den in [('phys__saturation_variation','phys__saturation_mean'),
                          ('phys__brightness_variation','phys__brightness_mean')]:
                if c in cols:x[:,cols.index(c)]=relative(x[:,cols.index(c)],numeric(frame[den]))
        return x


def tree_estimator(spec):
    if spec['learner']=='extra':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=80,max_depth=spec.get('depth'),
            min_samples_leaf=spec['leaf'],random_state=42,n_jobs=1)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=120,depth=spec['depth'],l2_leaf_reg=spec['l2'],
        learning_rate=.04,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)

SemanticHead = Head

"""Meaning-preserving head training. Creative coefficient remains unchanged."""
import numpy as np
from scipy.special import expit, logit, ndtr, ndtri
from scipy.stats import rankdata
PreviousHead = SemanticHead

class KernelSVR:
    """Additive per-feature response kernel, or pairwise interactions."""
    def __init__(self,spec):self.spec=spec
    def kernel(self,x,z):
        gamma=self.spec['gamma'];kind=self.spec.get('kernel','rbf')
        if kind=='laplace':return np.exp(-gamma*np.abs(x[:,None,:]-z[None,:,:]).sum(2))
        parts=np.exp(-gamma*(x[:,None,:]-z[None,:,:])**2)
        if kind=='additive':return parts.mean(2)
        if kind=='pairwise':
            n=x.shape[1];return (parts.sum(2)**2-(parts**2).sum(2))/(n*(n-1))
        raise ValueError(kind)
    def fit(self,x,y,sample_weight):
        from sklearn.svm import SVR
        self.x=x.copy();sp=self.spec
        self.estimator=SVR(kernel='precomputed',C=sp['C'],epsilon=sp.get('epsilon',.05))
        self.estimator.fit(self.kernel(x,x),y,sample_weight=sample_weight);return self
    def predict(self,x):return self.estimator.predict(self.kernel(x,self.x))

def svr_estimator(spec):
    if spec.get('bag_training'):return BagSVR(spec)
    if spec.get('kernel') in ['additive','pairwise','laplace']:return KernelSVR(spec)
    from sklearn.svm import SVR
    return SVR(C=spec['C'],gamma=spec['gamma'],epsilon=spec.get('epsilon',.05),kernel=spec.get('kernel','rbf'))

class BagSVR:
    """Fit one human target to the mean response across its cached feature readings."""
    def __init__(self,spec):self.spec=spec
    def kernel(self,x,z):
        from scipy.spatial.distance import cdist
        return np.exp(-self.spec['gamma']*cdist(x,z,'sqeuclidean'))
    def fit(self,x,y,sample_weight,groups):
        from sklearn.svm import SVR
        _,inv=np.unique(groups,return_inverse=True);n=inv.max()+1
        matrix=np.eye(n)[inv];self.avg=matrix/matrix.sum(0);self.x=x.copy()
        target=self.avg.T@y;w=matrix.T@sample_weight
        k=self.avg.T@self.kernel(x,x)@self.avg
        self.estimator=SVR(kernel='precomputed',C=self.spec['C'],epsilon=self.spec.get('epsilon',.05))
        self.estimator.fit(k,target,sample_weight=w);return self
    def predict(self,x):return self.estimator.predict(self.kernel(x,self.x)@self.avg)

def train_estimator(estimator,x,y,w,records,spec):
    if spec.get('bag_training'):return estimator.fit(x,y,sample_weight=w,groups=records)
    return estimator.fit(x,y,sample_weight=w)

def fit_calibration(pred, raw, records, kind):
    import pandas as pd
    d=pd.DataFrame(dict(record=records,pred=pred,raw=raw)).groupby('record').mean()
    x=d.pred.to_numpy();y=d.raw.to_numpy()
    if kind=='moments':
        slope=y.std()/max(x.std(),1e-8);return dict(kind='linear',slope=float(slope),intercept=float(y.mean()-slope*x.mean()))
    if kind=='linear':
        slope=max(0,float(np.cov(x,y,ddof=0)[0,1]/max(x.var(),1e-8)))
        return dict(kind='linear',slope=slope,intercept=float(y.mean()-slope*x.mean()))
    if kind=='isotonic':
        from sklearn.isotonic import IsotonicRegression
        ir=IsotonicRegression(out_of_bounds='clip').fit(x,y)
        return dict(kind='interp',x=ir.X_thresholds_.tolist(),y=ir.y_thresholds_.tolist())
    if kind=='quantile':
        levels=np.linspace(0,1,11);a=np.quantile(x,levels);b=np.quantile(y,levels)
        uniq,ix=np.unique(a,return_index=True)
        return dict(kind='interp',x=uniq.tolist(),y=b[ix].tolist())
    raise ValueError(kind)

def apply_calibration(pred,cal):
    if cal['kind']=='linear':pred=pred*cal['slope']+cal['intercept']
    else:pred=np.interp(pred,cal['x'],cal['y'])
    return np.clip(pred,.001,1.)

def target_forward(raw, spec, records=None):
    kind=spec['target']; cal=raw
    if kind=='logit': y=logit(np.clip(raw,.001,.999))
    elif kind=='probit': y=ndtri(np.clip(raw,.001,.999))
    elif kind in ('rank','rank_records','normal_rank','normal_rank_records'):
        if 'records' in kind:
            _,first,inverse=np.unique(records,return_index=True,return_inverse=True)
            cal=raw[first]; y=(rankdata(cal)/(len(cal)+1))[inverse]
        else:y=rankdata(raw)/(len(raw)+1)
        if kind.startswith('normal'):y=ndtri(y)
    elif kind=='log':y=np.log(np.maximum(raw,.001))
    else:y=raw.copy()
    return y,cal

def target_inverse(p,spec,cal):
    kind=spec['target']
    if kind=='logit':p=expit(p)
    elif kind=='probit':p=ndtr(p)
    elif kind=='log':p=np.exp(np.clip(p,-12,3))
    elif kind in ('rank','rank_records','normal_rank','normal_rank_records'):
        if kind.startswith('normal'):p=ndtr(p)
        p=np.quantile(cal,np.clip(p,0,1))
    return np.clip(p,.001,1.)

class Head(PreviousHead):
    @classmethod
    def fit(cls,train,spec):
        if spec.get('ensemble_members'):
            members=[cls.fit(train,sp) for sp in spec['ensemble_members']]
            return cls(dict(spec=spec,columns=spec['fixed'],training_families=sorted(train.family.unique())),members)
        if spec.get('calibration'):
            inner=dict(spec);kind=inner.pop('calibration')
            obj=cls.fit(train,inner);p=obj.predict(train)
            obj.state['calibration']=fit_calibration(p,train.y.to_numpy(),train.record.to_numpy(),kind)
            obj.state['spec']=spec
            return obj
        if not spec.get('new_training'):return super().fit(train,spec)
        # Reuse established, fold-only input preprocessing; fit the new target model.
        obj=super().fit(train,dict(spec,target='raw',learner='ridge',alpha=100))
        obj.state['spec']=spec
        z=obj.design(train); raw=train.y.to_numpy(); y,cal=target_forward(raw,spec,train.record.to_numpy())
        mu=y.mean();scale=max(y.std(),1e-8); y=(y-mu)/scale
        w=1/train.groupby('family').family.transform('size').to_numpy()
        if 'avito_weight' in spec:
            total=w.sum();w*=np.where(train.brand.eq('Avito'),spec['avito_weight'],1.);w*=total/w.sum()
        if spec['learner']=='ridge':
            a=np.column_stack([np.ones(len(z)),z]);pen=np.eye(a.shape[1])*spec['alpha'];pen[0,0]=0
            estimator=np.linalg.solve(a.T@(w[:,None]*a)+pen,a.T@(w*y))
        else:
            estimator=svr_estimator(spec)
            train_estimator(estimator,z,y,w,train.record.to_numpy(),spec)
        obj.estimator=estimator
        obj.state.update(target_mean=float(mu),target_scale=float(scale),raw_y=cal.tolist())
        return obj

    def predict(self,frame):
        sp=self.state['spec']
        if sp.get('ensemble_members'):return np.mean([m.predict(frame) for m in self.estimator],axis=0)
        if 'calibration' in self.state:
            state=dict(self.state);cal=state.pop('calibration');inner=dict(sp);inner.pop('calibration');state['spec']=inner
            return apply_calibration(type(self)(state,self.estimator).predict(frame),cal)
        if not sp.get('new_training'):return super().predict(frame)
        z=self.design(frame)
        p=np.column_stack([np.ones(len(z)),z])@self.estimator if sp['learner']=='ridge' else self.estimator.predict(z)
        p=p*self.state['target_scale']+self.state['target_mean']
        return target_inverse(p,sp,np.array(self.state['raw_y']))

NoticeabilityHead = Head

"""MD regression variants; fitted only within the current family exclusion fold."""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.svm import SVR
PriorHead = SemanticHead

def reliability_weights(w,noticeability,power):
    if not power:return w
    n=np.asarray(noticeability,float);n=np.where(np.isfinite(n),n,np.nanmedian(n));n=np.maximum(n,1e-6)**power
    result=w*n;return result*w.sum()/result.sum()

def transform(y,target):
    if target=='log':return np.log(np.maximum(y,.001))
    if target=='sqrt':return np.sqrt(np.maximum(y,0))
    if target=='logit':return logit(np.clip(y,.001,.999))
    if target=='rank':return rankdata(y)/(len(y)+1)
    return np.asarray(y).copy()

def invert(p,target,raw_y):
    if target=='log':return np.exp(np.clip(p,-12,3))
    if target=='sqrt':return np.maximum(p,0)**2
    if target=='logit':return expit(p)
    if target=='rank':return np.quantile(raw_y,np.clip(p,0,1))
    return p

def regress(x,xp,y0,w,spec):
    y=transform(y0,spec['target']);mu=y.mean();sd=max(y.std(),1e-8);z=(y-mu)/sd
    if spec['learner']=='ridge':
        a=np.column_stack([np.ones(len(x)),x]);pen=np.eye(a.shape[1])*spec['alpha'];pen[0,0]=0
        model=np.linalg.solve(a.T@(w[:,None]*a)+pen,a.T@(w*z))
        train=a@model;pred=np.column_stack([np.ones(len(xp)),xp])@model
    else:
        model=SVR(C=spec['C'],gamma=spec['gamma'],epsilon=spec.get('epsilon',.05));model.fit(x,z,sample_weight=w)
        train=model.predict(x);pred=model.predict(xp)
    train=invert(train*sd+mu,spec['target'],y0);pred=invert(pred*sd+mu,spec['target'],y0)
    intercept=0.;slope=1.
    if spec.get('calibration')=='train_affine':
        mx=np.average(train,weights=w);my=np.average(y0,weights=w)
        slope=max(0.,np.average((train-mx)*(y0-my),weights=w)/max(np.average((train-mx)**2,weights=w),1e-12))
        intercept=my-slope*mx;pred=intercept+slope*pred
    return np.clip(pred,.001,1.),model,mu,sd,intercept,slope

class Head(PriorHead):
    def raw(self,frame):
        x=super().raw(frame)
        if self.state['spec'].get('panel_consensus'):
            for j,c in enumerate(self.state['columns']):
                if c.startswith(('panel__','fresh__')):x[:,j]=np.floor(x[:,j]+.5)
        return x

    @classmethod
    def fit(cls,train,spec):
        if not spec.get('md_refinement'):return super().fit(train,spec)
        state=dict(spec=spec,columns=spec['fixed'],history=train[['family','brand','vertical','y']].to_dict('records'),training_families=sorted(train.family.unique()))
        obj=cls(state,None);x=obj.raw(train)
        med=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in x.T]);x=np.where(np.isfinite(x),x,med)
        center=x.mean(0);scale=x.std(0);scale[scale<1e-9]=1.
        state.update(median=med.tolist(),center=center.tolist(),scale=scale.tolist())
        x=obj.design(train);w=1/train.groupby('family').family.transform('size').to_numpy();y=train.y.to_numpy()
        w=reliability_weights(w,train.noticeability_percent.to_numpy(),spec.get('noticeability_weight_power',0))
        _,model,mu,sd,a,b=regress(x,x,y,w,spec)
        state.update(raw_y=y.tolist(),target_mean=float(mu),target_scale=float(sd),calibration_intercept=float(a),calibration_slope=float(b))
        obj.estimator=model;return obj

    def predict(self,frame):
        spec=self.state['spec']
        if not spec.get('md_refinement'):return super().predict(frame)
        z=self.design(frame);st=self.state
        p=np.column_stack([np.ones(len(z)),z])@self.estimator if spec['learner']=='ridge' else self.estimator.predict(z)
        p=invert(p*st['target_scale']+st['target_mean'],spec['target'],st['raw_y'])
        return np.clip(st['calibration_intercept']+st['calibration_slope']*p,.001,1.)

MessageHead = Head

"""Equal voting of two recall estimators using a shared seven-input schema."""
import numpy as np
RecallPriorHead = SemanticHead

class Head(RecallPriorHead):
    @classmethod
    def fit(cls,train,spec):
        if 'members' not in spec:return super().fit(train,spec)
        models=[RecallPriorHead.fit(train,member) for member in spec['members']]
        state=dict(spec=spec,columns=spec['fixed'],training_families=sorted(train.family.unique()))
        return cls(state,[(m.state,m.estimator) for m in models])
    def predict(self,frame):
        if 'members' not in self.state['spec']:return super().predict(frame)
        return np.mean([RecallPriorHead(state,estimator).predict(frame) for state,estimator in self.estimator],axis=0)

RecallHead = Head

class Head(SemanticHead):
    @classmethod
    def fit(cls,train,spec):
        kind=NoticeabilityHead if spec['task']=='n' else MessageHead if spec['task']=='m' else RecallHead
        return kind.fit(train,spec)
    @classmethod
    def load(cls,path):
        path=Path(path);state=json.loads((path/'state.json').read_text())
        task=state['spec']['task'];kind=NoticeabilityHead if task=='n' else MessageHead if task=='m' else RecallHead
        return kind(state,joblib.load(path/'estimator.joblib'))

PreviousNoticeabilityHead = NoticeabilityHead
class NoticeabilityHead(PreviousNoticeabilityHead):
    def raw(self,frame):
        columns=self.state['spec'].get('time_exposure_share',[])
        if columns:
            frame=frame.copy()
            for column in columns:
                frame[column]=numeric(frame[column])/np.maximum(1,numeric(frame['phys__duration']))
        return super().raw(frame)

"""Same recall inputs, consistent aggregation of repeated observed judgments."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.svm import SVR
from sklearn.preprocessing import PowerTransformer
BaseHead = OriginalHead

def recall_transform(y,spec):
    target=spec['target'];extra={}
    if target=='log':value=np.log(np.maximum(y,.001))
    elif target=='sqrt':value=np.sqrt(np.maximum(y,.001))
    elif target=='log1p':value=np.log1p(y)
    elif target=='rank':value=rankdata(y)/(len(y)+1)
    elif target=='boxcox':
        pt=PowerTransformer(method='box-cox',standardize=False).fit(np.maximum(y,.001)[:,None])
        value=pt.transform(np.maximum(y,.001)[:,None])[:,0];extra['lambda']=float(pt.lambdas_[0])
    else:value=y.copy()
    return value,extra

def recall_inverse(p,spec,extra):
    target=spec['target']
    if target=='log':return np.exp(np.clip(p,-12,3))
    if target=='sqrt':return np.maximum(p,0)**2
    if target=='log1p':return np.expm1(p)
    if target=='rank':return np.quantile(extra['raw_y'],np.clip(p,0,1))
    if target=='boxcox':
        lam=extra['lambda']
        if abs(lam)<1e-8:return np.exp(np.clip(p,-12,3))
        return np.maximum(1e-9,lam*p+1)**(1/lam)
    return p

class Head(BaseHead):
    def raw(self,frame):
        spec=self.state['spec'];data=frame
        cols=spec.get('consensus_features',[])
        if cols or spec.get('integer_characters'):
            data=frame.copy();group='sha' if 'sha' in data else 'record'
            for col in cols:
                values=data.groupby(group)[col].transform('mean').to_numpy()
                rule=spec.get('consensus_rule','mean')
                if rule=='majority':values=np.where(np.isfinite(values),(values>.5).astype(float),np.nan)
                elif rule=='two_thirds':values=np.where(np.isfinite(values),(values>=2/3).astype(float),np.nan)
                data[col]=values
            if spec.get('integer_characters'):
                values=data.groupby(group).human_characters_count.transform('mean').to_numpy()
                data['human_characters_count']=np.floor(values+.5)
        return super().raw(data)

    @classmethod
    def fit(cls,train,spec):
        if 'members' in spec:
            heads=[cls.fit(train,sp) for sp in spec['members']]
            return cls(dict(spec=spec,columns=spec['fixed'],training_families=sorted(train.family.unique())),[(h.state,h.estimator) for h in heads])
        state=dict(spec=spec,columns=spec['fixed'],history=train[['family','brand','vertical','y']].to_dict('records'),training_families=sorted(train.family.unique()))
        result=cls(state,None);x=result.raw(train)
        med=np.nanmedian(x,axis=0);x=np.where(np.isfinite(x),x,med)
        center=x.mean(0);scale=x.std(0);scale[scale<1e-9]=1.
        if spec.get('scaler')=='robust':
            robust=(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0))/1.349
            scale=np.where(robust<1e-9,scale,robust)
        state.update(median=med.tolist(),center=center.tolist(),scale=scale.tolist())
        z=result.design(train);raw=train.y.to_numpy();y,extra=recall_transform(raw,spec)
        mu=y.mean();sd=max(y.std(),1e-8);y=(y-mu)/sd
        w=1/train.groupby('family').family.transform('size').to_numpy();total=w.sum()
        w*=np.where(train.brand.eq('Avito'),spec.get('avito_weight',1),1);w*=total/w.sum()
        model=SVR(C=spec['C'],gamma=spec['gamma'],epsilon=spec.get('epsilon',.05));model.fit(z,y,sample_weight=w)
        state.update(target_mean=float(mu),target_scale=float(sd),raw_y=raw.tolist(),**extra)
        result.estimator=model;return result

    def predict(self,frame):
        state=self.state;spec=state['spec']
        if 'members' in spec:return np.mean([type(self)(st,est).predict(frame) for st,est in self.estimator],axis=0)
        p=self.estimator.predict(self.design(frame))*state['target_scale']+state['target_mean']
        p=recall_inverse(p,spec,state)
        return np.clip(p,.001,max(1,max(state['raw_y'])*1.2))

RecallHead = Head

class Head(SemanticHead):
    @classmethod
    def fit(cls,train,spec):
        kind=NoticeabilityHead if spec['task']=='n' else MessageHead if spec['task']=='m' else RecallHead
        return kind.fit(train,spec)
    @classmethod
    def load(cls,path):
        path=Path(path);state=json.loads((path/'state.json').read_text());task=state['spec']['task']
        kind=NoticeabilityHead if task=='n' else MessageHead if task=='m' else RecallHead
        return kind(state,joblib.load(path/'estimator.joblib'))
