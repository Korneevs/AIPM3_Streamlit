"""Finished-video means replace only production-sensitive neuromatics inputs."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

POLICY_DIR = Path(__file__).resolve().parent / 'neuromatics_policy' / '20261008'
SCORING_VERSION = 'neuromatics-finished-means-20261008'


def artifact_hashes():
    manifest = json.loads((POLICY_DIR / 'manifest.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((POLICY_DIR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Neuromatics mean policy mismatch: {name}')
    return manifest


def load_policy():
    artifact_hashes()
    result = json.loads((POLICY_DIR / 'policy.json').read_text())
    if result['scoring_version'] != SCORING_VERSION:
        raise ValueError('Mean policy belongs to a different scoring version')
    return result


class FixedMeanHead:
    """Keep fitted weights; replace selected inputs after normalization/consensus."""

    def __init__(self, base, constants):
        self.base = base
        self.state = base.state
        self.constants = constants
        self.neutralized_features = frozenset(constants)
        self._indices = {feature: self.state['columns'].index(feature) for feature in constants}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def raw(self, frame):
        values = self.base.raw(frame).copy()
        for feature, index in self._indices.items():
            values[:, index] = self.constants[feature]['model_input_mean']
        return values

    def _fixed_design(self, values):
        values = np.asarray(values, float).copy()
        for feature, index in self._indices.items():
            values[:, index] = self.constants[feature]['design_value']
        return values

    def design(self, frame):
        return self._fixed_design(self.base.design(frame))

    def predict_design(self, values):
        return self.base.predict_design(self._fixed_design(values))

    def predict(self, frame):
        return self.predict_design(self.design(frame))

    def observed_measurement(self, frame, feature):
        if feature not in self.constants:
            return self.base.observed_measurement(frame, feature)
        observed = float(pd.to_numeric(frame[feature], errors='coerce').mean())
        return self.constants[feature]['model_input_mean'], observed


def apply_policy(model, policy):
    for task, constants in policy['constants'].items():
        if constants:
            model.heads[task] = FixedMeanHead(model.heads[task], constants)
    return model
