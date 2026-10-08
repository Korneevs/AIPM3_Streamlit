"""Build the user-selected mean policy; no fitting and no external calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from aipm3 import latest_runtime as runtime
from aipm3.neuromatics_models import NeuromaticsModels
from aipm3.neuromatics_mean_policy import apply_policy, POLICY_DIR, SCORING_VERSION


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def build(root):
    out = root / 'outputs/AIPM_neuromatics_mean_policy_20261008'
    out.mkdir(exist_ok=True)
    old_dir = root / 'outputs/AIPM_neuromatics_expanded_20261002'
    repeats_path = root / 'outputs/AIPM_neuro_prompt_rerun_20261008/borderline_measurements.json'
    summary_path = root / 'outputs/AIPM_feature_pair_disagreement_20261008/feature_summary.csv'
    old = {t: pd.read_csv(old_dir / f'canonical_{t}.csv', float_precision='round_trip') for t in 'nmr'}
    pairs = pd.read_csv(old_dir / 'pair_scores.csv')
    summary = pd.read_csv(summary_path).set_index('feature')
    repeats = json.loads(repeats_path.read_text())
    new = pd.DataFrame([dict(pair_id=r['pair_id'], repeat=r['repeat'], **r['features']) for r in repeats])
    assert len(new) == 60 and new.groupby('pair_id').size().eq(5).all()
    assert all(sorted(g.repeat) == list(range(1, 6)) for _, g in new.groupby('pair_id'))
    frozen = runtime.load_latest_models()
    base = NeuromaticsModels(frozen, runtime.model_module().coefficient, use_mean_policy=False)
    decisions = []
    for task in 'nmr':
        for feature in frozen.heads[task].state['columns']:
            if feature == 'brand_history':
                decisions.append(dict(task=task, feature=feature, before=None, after=None, action='keep', reason='Historical context, not inferred from video'))
                continue
            count = int(summary.loc[feature, 'mismatch_pairs'])
            after = None
            if 5 <= count <= 7:
                mismatches = []
                for pair in pairs.itertuples():
                    value = float(new.loc[new.pair_id.eq(pair.pair_id), feature].mean())
                    reference = float(old[task].loc[old[task].sha.eq(pair.finished_sha), feature].mean())
                    different = (value >= 2/3) != (reference >= 2/3) if task == 'r' else abs(value-reference) >= 1-1e-10
                    if different:
                        mismatches.append(pair.pair_id)
                after = len(mismatches)
            action = 'keep' if count <= 4 else 'keep_neuromatics_prompt' if 5 <= count <= 7 and after <= 4 else 'finished_mean'
            decisions.append(dict(task=task, feature=feature, label=summary.loc[feature, 'label'],
                                  before=count, after=after, action=action))
    constants = {t: {} for t in 'nmr'}
    sources = {}
    for task in 'nmr':
        path = runtime.BUNDLE_DIR / 'data' / f'fit_{task}.csv'
        train = pd.read_csv(path, float_precision='round_trip')
        keys = train['sha'].where(train['sha'].notna(), 'record:' + train.record.astype(str))
        sources[task] = dict(path=str(path.relative_to(APP)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             rows=len(train), distinct_videos=int(keys.nunique()))
        raw = pd.DataFrame(frozen.heads[task].raw(train), columns=frozen.heads[task].state['columns']) if task == 'n' else train
        for decision in [d for d in decisions if d['task'] == task and d['action'] == 'finished_mean']:
            feature = decision['feature']
            per_video = pd.to_numeric(raw[feature], errors='raise').groupby(keys).mean().dropna()
            mean = float(per_video.mean())
            head = base.heads[task]
            state = head.state if task == 'n' else head.parameters
            index = state['columns'].index(feature) if task == 'n' else state['features'].index(feature)
            transformed = mean
            if 'clip_lo' in state:
                lo, hi = state['clip_lo'][index], state['clip_hi'][index]
                transformed = float(np.clip(transformed, -np.inf if lo is None else lo, np.inf if hi is None else hi))
            design = (transformed-state['center'][index])/state['scale'][index]
            step = state.get('spec', {}).get('step', 0)
            if isinstance(step, (int, float)) and step:
                design = float(np.round(design/step)*step)
            constants[task][feature] = dict(model_input_mean=mean, design_value=float(design),
                video_count=len(per_video), source_task=task,
                scale='fraction_of_duration' if feature in head.state['spec'].get('time_exposure_share', []) else 'original_feature_scale',
                observed_mean=float(pd.to_numeric(train[feature]).groupby(keys).mean().mean()))
    policy = dict(scoring_version=SCORING_VERSION, rule='Keep <=4/12; keep 5-7/12 only if new <=4/12; otherwise finished mean',
        pair_count=12, new_repeats=5, old_repeats=10, weighting='Equal weight per distinct finished video; average its repeats first',
        constants=constants, decisions=decisions, sources=sources,
        fitted_weights_changed=False, finished_model_changed=False,
        evaluation='Same 12 pairs used to choose features; descriptive diagnostic, not independent validation',
        selection_inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [repeats_path, summary_path]})
    selected = apply_policy(NeuromaticsModels(frozen, runtime.model_module().coefficient, use_mean_policy=False), policy)
    rows = []; frames_out = {t: [] for t in 'nmr'}
    for pair in pairs.itertuples():
        row = dict(pair_id=pair.pair_id, cohort=pair.cohort_origin)
        for label, model, sha, count in [('before10', base, pair.prototype_sha, 10), ('before5', base, pair.prototype_sha, 5),
                                         ('after5', selected, pair.prototype_sha, 5), ('finished', frozen, pair.finished_sha, 10)]:
            pred = {}
            for task in 'nmr':
                frame = old[task].loc[old[task].sha.eq(sha)&old[task].repeat.le(count)].sort_values('repeat').copy()
                assert len(frame) == count
                if label == 'after5':
                    observations = new.loc[new.pair_id.eq(pair.pair_id)].set_index('repeat')
                    for decision in decisions:
                        if decision['task'] == task and decision['after'] is not None:
                            frame[decision['feature']] = frame.repeat.map(observations[decision['feature']])
                    frames_out[task].append(frame)
                pred[task] = model.heads[task].predict(frame)
                row[label+'_'+task] = float(pred[task].mean())
            row[label+'_Q'] = float((pred['n']*pred['m']*pred['r']).mean())
        rows.append(row)
    scores = pd.DataFrame(rows)
    metrics = {cohort: {label: {task: float(spearmanr(scores.loc[mask, label+'_'+task], scores.loc[mask, 'finished_'+task]).statistic)
        for task in ['n', 'm', 'r', 'Q']} for label in ['before10', 'before5', 'after5']}
        for cohort, mask in [('all12', np.ones(len(scores), bool)), ('original7', scores.cohort.eq('original_7')), ('added5', scores.cohort.ne('original_7'))]}
    calibration = json.loads((runtime.MESSAGE_BUNDLE_DIR / 'neuromatics_display_reference.json').read_text())
    for task in ['n', 'r', 'Q']:
        factor = float(np.exp(np.median(np.log(scores['finished_'+task]/scores['after5_'+task]))))
        reference = calibration['references'][task]['finished_reference']
        calibration['references'][task] = dict(mean=reference['mean']/factor, cuts=[v/factor for v in reference['cuts']],
                                               finished_scale_factor=factor, finished_reference=reference)
    calibration.update(version='neuromatics-means-display-pairs12-20261008', scoring_version=SCORING_VERSION,
                       validation='Same-pair diagnostic mapping; not independent class validation')
    dump(POLICY_DIR / 'policy.json', policy)
    dump(POLICY_DIR / 'display_reference.json', calibration)
    dump(POLICY_DIR / 'manifest.json', {name: hashlib.sha256((POLICY_DIR/name).read_bytes()).hexdigest()
         for name in ['policy.json', 'display_reference.json']})
    dump(out / 'policy.json', policy); dump(out / 'metrics.json', metrics)
    scores.to_csv(out / 'pair_scores.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'feature_decisions.csv', index=False)
    for task, frames in frames_out.items():
        pd.concat(frames).to_csv(out / f'inputs_{task}.csv', index=False)
    print(json.dumps(metrics, indent=2))
    print('Replaced:', {t:list(v) for t,v in constants.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workspace', type=Path, required=True)
    build(parser.parse_args().workspace.resolve())
