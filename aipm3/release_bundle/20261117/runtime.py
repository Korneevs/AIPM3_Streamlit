"""Portable inference for the frozen latest AIPM .719 model.

No fitting, network requests, source checkouts, or writes occur at inference.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent
SCORING_VERSION = 'aipm-0719-refined-20261117'


def _json(path):
    return json.loads(Path(path).read_text())


def artifact_hashes(bundle_dir=None):
    path = Path(bundle_dir) if bundle_dir else BUNDLE_DIR
    hashes = _json(path / 'artifacts.json')
    for name, expected in hashes.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError('AIPM .719 artifact mismatch: ' + name)
    return hashes


def score_classes(scores, bundle_dir=None):
    path = Path(bundle_dir) if bundle_dir else BUNDLE_DIR
    thresholds = _json(path / 'manifest.json')['class_thresholds']
    names = dict(n='noticeability', m='message_delivery', r='norm_ad_recall', q='Q')
    return {task: int(np.searchsorted(thresholds[task], float(scores.get(task, scores.get(name))), side='left') + 1)
            for task, name in names.items()}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


native = _module('_aipm0719_native', BUNDLE_DIR / 'native_model.py')
neuro_source = _module('_aipm0719_neuro_reference', BUNDLE_DIR / 'neuro_recall.py')


def public_identifier(value, prefix):
    if pd.isna(value):
        return value
    value = str(value)
    if value.startswith(prefix + '_') and len(value) == 66:
        try:
            int(value[2:], 16)
            return value
        except ValueError:
            pass
    return prefix + '_' + hashlib.sha256((prefix + ':' + value).encode()).hexdigest()


def _public_frame(frame):
    frame = frame.copy()
    frame['family'] = frame.family.map(lambda value: public_identifier(value, 'f'))
    return frame


def family_for_video(source_sha, bundle_dir=None):
    path = Path(bundle_dir) if bundle_dir else BUNDLE_DIR
    query = public_identifier(source_sha, 's')
    families = set()
    for task in 'nmr':
        frame = pd.read_csv(path / 'data' / ('fit_' + task + '.csv'), usecols=['sha', 'family'])
        families.update(frame.loc[frame.sha.eq(query), 'family'].dropna())
    if len(families) > 1:
        raise ValueError('Conflicting pseudonymous family groups')
    return next(iter(families)) if families else None


class NativeLeaf:
    def __init__(self, model, task):
        self.model, self.task, self.state = model, task, model.state
        self.neutralized_features = frozenset()

    def _members(self):
        return [NativeLeaf(native.RecallHead(state, estimator), 'r')
                for state, estimator in self.model.estimator]

    def raw(self, frame):
        if 'members' not in self.state['spec']:
            return self.model.raw(_public_frame(frame))
        values = {}
        for member in self._members():
            for name, column in zip(member.state['columns'], member.raw(frame).T):
                if name in values:
                    np.testing.assert_allclose(values[name], column, equal_nan=True)
                values[name] = column
        return np.column_stack([values[name] for name in self.state['columns']])

    def design(self, frame):
        if 'members' in self.state['spec']:
            raise ValueError('Use attribution_components for member-specific designs')
        return self.model.design(_public_frame(frame))

    def predict_design(self, values):
        state, spec = self.state, self.state['spec']
        if 'members' in spec:
            raise ValueError('Use attribution_components for member-specific designs')
        values = np.asarray(values, dtype=float)
        prediction = (np.column_stack([np.ones(len(values)), values]) @ self.model.estimator
                      if spec['learner'] == 'ridge' else self.model.estimator.predict(values))
        prediction = prediction * state['target_scale'] + state['target_mean']
        if self.task == 'n':
            return native.target_inverse(prediction, spec, np.asarray(state['raw_y']))
        prediction = native.recall_inverse(prediction, spec, state)
        return np.clip(prediction, .001, max(1., max(state['raw_y']) * 1.2))

    def predict(self, frame):
        return self.model.predict(_public_frame(frame))

    def attribution_components(self, frame, background):
        if 'members' in self.state['spec']:
            members = self._members()
            return [dict(item, weight=item['weight'] / len(members))
                    for member in members for item in member.attribution_components(frame, background)]
        return [dict(weight=1., columns=self.state['columns'], x=self.design(frame),
                     background=self.design(background), predict=self.predict_design)]

    def observed_measurement(self, frame, feature):
        original = float(pd.to_numeric(frame[feature], errors='coerce').mean()) if feature in frame else float('nan')
        consensus = self.state['spec'].get('consensus_features', [])
        if 'members' in self.state['spec']:
            consensus = set().union(*(s.get('consensus_features', []) for s in self.state['spec']['members']))
        used = float(self.raw(frame)[:, self.state['columns'].index(feature)].mean()) if feature == 'brand_history' or feature in consensus else original
        return used, original


class MessageLeaf:
    def __init__(self, path):
        self.__dict__.update(joblib.load(Path(path) / 'model.joblib'))
        self.pre = native.Head.load(Path(path) / 'preprocessor')
        self.neutralized_features = frozenset()
        assert self.spec['aggregation'] == 'repeats' and self.poly is None

    def raw(self, frame):
        return self.pre.raw(frame)

    def design(self, frame):
        return self.pre.design(frame) * self.weights

    def predict_design(self, values):
        prediction = self.model.predict(np.asarray(values, dtype=float)) * self.sd + self.mu
        prediction = native.invert(prediction, self.spec['target'], self.cal_raw)
        return np.clip(self.intercept + self.slope * prediction, .001, 1.)

    def predict(self, frame):
        return self.predict_design(self.design(frame))

    def attribution_components(self, frame, background):
        return [dict(weight=1., columns=self.state['columns'], x=self.design(frame),
                     background=self.design(background), predict=self.predict_design)]

    def observed_measurement(self, frame, feature):
        value = float(pd.to_numeric(frame[feature], errors='coerce').mean())
        return value, value


class NeuroNoticeability(NativeLeaf):
    def __init__(self, model, settings, constants):
        super().__init__(model, 'n')
        self.settings, self.constants = settings, constants
        self.neutralized_features = frozenset(constants)

    def transformed(self, frame):
        result = frame.copy(deep=True)
        physical = result[self.settings['physical_columns']].to_numpy(float)
        state = result[self.settings['state_source']].to_numpy(float)
        if not np.isfinite(physical).all() or not np.isin(state, [0., 1.]).all():
            raise ValueError('Invalid raw neuro physical/state measurements')
        result.loc[:, self.settings['physical_columns']] = np.clip(physical, self.settings['lower'], self.settings['upper'])
        result[self.settings['state_input']] = state
        return result

    def _fixed(self, values, field):
        values = np.asarray(values, dtype=float).copy()
        for feature, point in self.constants.items():
            values[:, self.state['columns'].index(feature)] = point[field]
        return values

    def raw(self, frame):
        return self._fixed(self.model.raw(self.transformed(frame)), 'model_input_mean')

    def design(self, frame):
        return self._fixed(self.model.design(self.transformed(frame)), 'design_value')

    def predict_design(self, values):
        return super().predict_design(self._fixed(values, 'design_value'))

    def predict(self, frame):
        return self.predict_design(self.design(frame))

    def observed_measurement(self, frame, feature):
        original = float(pd.to_numeric(frame[feature], errors='coerce').mean()) if feature in frame else float('nan')
        if feature in self.constants:
            return self.constants[feature]['model_input_mean'], original
        transformed = self.transformed(frame)
        used = float(pd.to_numeric(transformed[feature], errors='coerce').mean()) if feature in transformed else original
        return used, original


class NeuroRecall:
    def __init__(self, parameters, constants):
        self.parameters = parameters
        self.constants = constants
        self.neutralized_features = frozenset(constants)
        self.adapter = object.__new__(neuro_source.NeuroRecall)
        self.adapter.state = parameters
        self.state = dict(columns=parameters['features'], history=parameters['history'],
                          spec=dict(task='r', learner='neuro_ridge', target='log',
                                    consensus_features=parameters['binary_consensus_features']))

    def _raw(self, frame):
        frame = _public_frame(frame)
        parts = []
        group = 'sha' if 'sha' in frame else 'record'
        for feature in self.state['columns']:
            if feature == 'brand_history':
                value = self.adapter._brand_history(frame)
            else:
                value = pd.to_numeric(frame[feature], errors='coerce').to_numpy(float)
                if feature == 'brand_first_mention_seconds':
                    value = value / np.maximum(1, pd.to_numeric(frame.total_video_duration_sec, errors='coerce').to_numpy(float))
                if feature in self.parameters['binary_consensus_features']:
                    value = frame.assign(_v=value).groupby(group)._v.transform('mean').to_numpy()
                    value = np.where(np.isfinite(value), (value >= self.parameters['binary_consensus_threshold']).astype(float), np.nan)
            parts.append(value)
        return np.column_stack(parts)

    def _fixed(self, values, field):
        values = np.asarray(values, dtype=float).copy()
        for feature, point in self.constants.items():
            values[:, self.state['columns'].index(feature)] = point[field]
        return values

    def raw(self, frame):
        return self._fixed(self._raw(frame), 'model_input_mean')

    def design(self, frame):
        p = self.parameters
        raw = self._raw(frame)
        raw = np.where(np.isfinite(raw), raw, np.asarray(p['median']))
        return self._fixed((raw - np.asarray(p['center'])) / np.asarray(p['scale']), 'design_value')

    def predict_design(self, values):
        p = self.parameters
        values = self._fixed(values, 'design_value')
        prediction = values @ np.asarray(p['coef']) + p['intercept']
        return np.exp(np.clip(prediction * p['target_sd'] + p['target_mu'], *p['log_output_clip']))

    def predict(self, frame):
        return self.predict_design(self.design(frame))

    def attribution_components(self, frame, background):
        return [dict(weight=1., columns=self.state['columns'], x=self.design(frame),
                     background=self.design(background), predict=self.predict_design)]

    def observed_measurement(self, frame, feature):
        original = float(pd.to_numeric(frame[feature], errors='coerce').mean()) if feature in frame else float('nan')
        if feature in self.constants:
            return self.constants[feature]['model_input_mean'], original
        if feature == 'brand_history' or feature in self.parameters['binary_consensus_features']:
            return float(self._raw(frame)[:, self.state['columns'].index(feature)].mean()), original
        return original, original


class Mixture:
    def __init__(self, leaves, weights, task):
        self.leaves, self.weights = leaves, weights
        self.state = deepcopy(leaves[0].state)
        self.state['spec'] = dict(task=task, learner='frozen_prediction_mixture', weights=weights)
        self.neutralized_features = frozenset.intersection(*(leaf.neutralized_features for leaf in leaves))

    def predict(self, frame):
        return sum(weight * leaf.predict(frame) for leaf, weight in zip(self.leaves, self.weights))

    def raw(self, frame):
        return sum(weight * leaf.raw(frame) for leaf, weight in zip(self.leaves, self.weights))

    def attribution_components(self, frame, background):
        return [dict(item, weight=weight * item['weight'])
                for leaf, weight in zip(self.leaves, self.weights)
                for item in leaf.attribution_components(frame, background)]

    def observed_measurement(self, frame, feature):
        points = [leaf.observed_measurement(frame, feature) for leaf in self.leaves]
        return sum(weight * point[0] for weight, point in zip(self.weights, points)), points[0][1]


class Models:
    def __init__(self, material_kind='finished', bundle_dir=None):
        self.path = Path(bundle_dir) if bundle_dir else BUNDLE_DIR
        artifact_hashes(self.path)
        self.manifest = _json(self.path / 'manifest.json')
        if self.manifest['scoring_version'] != SCORING_VERSION:
            raise ValueError('Unexpected scoring version')
        self.material_kind = material_kind
        if material_kind not in ('finished', 'neuromatics'):
            raise ValueError('Unknown material kind')
        self.heads = {}
        for task, mixture in self.manifest['mixtures'].items():
            leaves, weights = [], []
            for leaf_id, weight in mixture.items():
                folder = self.path / 'models' / leaf_id
                if task == 'm':
                    leaf = MessageLeaf(folder)
                elif task == 'n':
                    model = native.Head.load(folder)
                    if material_kind == 'neuromatics':
                        adapted = _json(folder / 'neuro.json')
                        leaf = NeuroNoticeability(model, _json(self.path / 'noticeability.json'), adapted['constants'])
                    else:
                        leaf = NativeLeaf(model, 'n')
                elif material_kind == 'neuromatics':
                    adapted = _json(folder / 'neuro.json')
                    leaf = NeuroRecall(adapted['parameters'], adapted['constants'])
                else:
                    payload = joblib.load(folder / 'head.joblib')
                    assert payload['kind'] in ('native', 'adapter')
                    leaf = NativeLeaf(native.RecallHead(payload['state'], payload['estimator']), 'r')
                leaves.append(leaf); weights.append(weight)
            self.heads[task] = Mixture(leaves, weights, task)
        self.coefficient = native.coefficient

    def attribution_background(self, task, frame):
        """Keep only valid raw neuro measurements in an explanation background."""
        result = frame.copy()
        if self.material_kind == 'neuromatics' and task == 'n':
            settings = _json(self.path / 'noticeability.json')
            valid = np.isfinite(result[settings['physical_columns']].to_numpy(float)).all(axis=1)
            valid &= result[settings['state_source']].isin([0., 1.]).to_numpy()
            result = result.loc[valid].copy()
        if result.empty:
            raise ValueError('No valid attribution background rows')
        return result

    def predict_repeats(self, noticeability, message_delivery, recall):
        frames = {t: pd.DataFrame(frame).sort_values(['record', 'repeat']).reset_index(drop=True)
                  for t, frame in zip('nmr', [noticeability, message_delivery, recall])}
        keys = frames['n'][['record', 'repeat']]
        if keys.empty or keys.duplicated().any() or not all(keys.equals(frame[['record', 'repeat']]) for frame in frames.values()):
            raise ValueError('Component repeat keys do not match')
        predictions = {t: self.heads[t].predict(frame) for t, frame in frames.items()}
        q = np.exp(sum(weight * np.log(predictions[task]) for task, weight in zip('nmr', self.manifest['q_exponents'])))
        return keys.assign(noticeability=predictions['n'], message_delivery=predictions['m'],
                           norm_ad_recall=predictions['r'], OPM=predictions['n'] * predictions['m'], Q=q)

    def score(self, noticeability, message_delivery, recall, reference_mean=None):
        repeated = self.predict_repeats(noticeability, message_delivery, recall)
        result = repeated.groupby('record', as_index=False)[['noticeability', 'message_delivery', 'norm_ad_recall', 'OPM', 'Q']].mean()
        if reference_mean is not None:
            result['C'] = self.coefficient(result.Q, reference_mean)
        return result

    def classify(self, task, values):
        return np.searchsorted(self.manifest['class_thresholds'][task], np.asarray(values, dtype=float), side='left') + 1


def load_models(material_kind='finished', bundle_dir=None):
    return Models(material_kind, bundle_dir)
