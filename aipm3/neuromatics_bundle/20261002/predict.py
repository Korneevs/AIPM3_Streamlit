"""Prediction-only neuro recall adapter. No fitting, APIs, or source-model imports.
Input rows: same seven feature meanings and metadata as original R. Exactly 10
repeats per video in production. Returns R for each row, preserving input order.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

class NeuroRecall:
    def __init__(self, parameter_path=None):
        path=Path(parameter_path) if parameter_path else Path(__file__).with_name('parameters.json')
        self.state=json.loads(path.read_text())

    def _brand_history(self, frame):
        h=pd.DataFrame(self.state['history']).copy()
        h['y']=np.log(np.maximum(h.y.to_numpy(float),.001))
        fm=h.groupby('family').y.mean().to_dict()
        brands={str(k):set(g.family) for k,g in h.groupby('brand',dropna=False)}
        verticals={str(k):set(g.family) for k,g in h.groupby('vertical',dropna=False)}
        total=sum(fm.values());out=[]
        for family,brand,vertical in zip(frame.family,frame.brand.astype(str),frame.vertical.astype(str)):
            global_mean=(total-fm.get(family,0))/(len(fm)-int(family in fm))
            vf=verticals.get(vertical,set())-{family}
            fallback=(sum(fm[f] for f in vf)+3*global_mean)/(len(vf)+3)
            bf=brands.get(brand,set())-{family}
            value=sum(fm[f] for f in bf)/len(bf) if bf else fallback
            out.append(value)
        return np.exp(out)

    def predict(self, frame):
        frame=pd.DataFrame(frame).copy();state=self.state;parts=[]
        group='sha' if 'sha' in frame else 'record'
        for feature in state['features']:
            if feature=='brand_history':value=self._brand_history(frame)
            else:
                value=pd.to_numeric(frame[feature],errors='coerce').to_numpy(float)
                if feature=='brand_first_mention_seconds':
                    duration=pd.to_numeric(frame.total_video_duration_sec,errors='coerce').to_numpy(float)
                    value=value/np.maximum(1,duration)
                if feature in state['binary_consensus_features']:
                    value=frame.assign(_v=value).groupby(group)._v.transform('mean').to_numpy()
                    value=np.where(np.isfinite(value),(value>=state['binary_consensus_threshold']).astype(float),np.nan)
            parts.append(value)
        x=np.column_stack(parts);x=np.where(np.isfinite(x),x,np.asarray(state['median']))
        z=(x-np.asarray(state['center']))/np.asarray(state['scale'])
        normalized=z@np.asarray(state['coef'])+state['intercept']
        log_r=normalized*state['target_sd']+state['target_mu']
        return np.exp(np.clip(log_r,*state['log_output_clip']))
