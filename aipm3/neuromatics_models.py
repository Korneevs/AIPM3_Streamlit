"""October 2 N/R adaptation plus the shared eight-input message head.

The finished-video package remains the source for N/M and the creative link.
This module performs no fitting, extraction, network requests or file writes.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE_DIR = Path(__file__).resolve().parent / "neuromatics_bundle" / "20261002"


def artifact_hashes():
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((BUNDLE_DIR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Neuromatics artifact mismatch: {name}")
    return manifest


@lru_cache(maxsize=1)
def _recall_module():
    artifact_hashes()
    spec = importlib.util.spec_from_file_location("_aipm_neuromatics_recall_20261002", BUNDLE_DIR / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoticeabilityHead:
    def __init__(self, frozen):
        self.frozen = frozen
        self.state = deepcopy(frozen.state)
        self.settings = json.loads((BUNDLE_DIR / "noticeability.json").read_text())

    def transformed(self, frame):
        data = frame.copy(deep=True)
        settings = self.settings
        required = [*settings["physical_columns"], settings["state_source"]]
        if set(required) - set(data):
            raise ValueError("Для нейроматика нужны исходные измерения заметности AIPM1.0.")
        physical = data[settings["physical_columns"]].to_numpy(float)
        state = data[settings["state_source"]].to_numpy(float)
        if not np.isfinite(physical).all() or not np.isin(state, [0., 1.]).all():
            raise ValueError("Некорректные измерения заметности нейроматика.")
        data.loc[:, settings["physical_columns"]] = np.clip(physical, settings["lower"], settings["upper"])
        data[settings["state_input"]] = state
        return data

    def raw(self, frame):
        return self.frozen.raw(self.transformed(frame))

    def design(self, frame):
        return self.frozen.design(self.transformed(frame))

    def predict(self, frame):
        return self.frozen.predict(self.transformed(frame))

    def predict_design(self, z):
        # The frozen N head is an SVR with its empirical-rank inverse transform.
        s = self.state
        assert s["spec"]["learner"] == "svr" and s["spec"]["target"] == "rank"
        p = self.frozen.estimator.predict(np.asarray(z, float)) * s["target_scale"] + s["target_mean"]
        return np.clip(np.quantile(s["raw_y"], np.clip(p, 0, 1)), .001, 1.)

    def observed_measurement(self, frame, feature):
        original = float(pd.to_numeric(frame[feature], errors="coerce").mean()) if feature in frame else float("nan")
        transformed = self.transformed(frame)
        used = float(pd.to_numeric(transformed[feature], errors="coerce").mean()) if feature in transformed else original
        return used, original


class RecallHead:
    def __init__(self):
        self.adapter = _recall_module().NeuroRecall(BUNDLE_DIR / "parameters.json")
        self.parameters = self.adapter.state
        self.state = {
            "columns": self.parameters["features"],
            "spec": {"task": "r", "learner": "neuro_ridge", "target": "log",
                     "consensus_features": self.parameters["binary_consensus_features"],
                     "consensus_rule": "two_thirds"},
            "history": self.parameters["history"],
        }

    def raw(self, frame):
        data = frame.copy()
        group = "sha" if "sha" in data else "record"
        values = []
        for feature in self.state["columns"]:
            if feature == "brand_history":
                value = self.adapter._brand_history(data)
            else:
                value = pd.to_numeric(data[feature], errors="coerce").to_numpy(float)
                if feature == "brand_first_mention_seconds":
                    value = value / np.maximum(1, pd.to_numeric(data.total_video_duration_sec, errors="coerce").to_numpy(float))
                if feature in self.parameters["binary_consensus_features"]:
                    value = data.assign(_v=value).groupby(group)._v.transform("mean").to_numpy()
                    value = np.where(np.isfinite(value), (value >= self.parameters["binary_consensus_threshold"]).astype(float), np.nan)
            values.append(value)
        return np.column_stack(values)

    def design(self, frame):
        p = self.parameters
        x = self.raw(frame)
        x = np.where(np.isfinite(x), x, np.asarray(p["median"]))
        return (x - np.asarray(p["center"])) / np.asarray(p["scale"])

    def predict_design(self, z):
        p = self.parameters
        normalized = np.asarray(z, float) @ np.asarray(p["coef"]) + p["intercept"]
        log_r = normalized * p["target_sd"] + p["target_mu"]
        return np.exp(np.clip(log_r, *p["log_output_clip"]))

    def predict(self, frame):
        return self.adapter.predict(frame)

    def observed_measurement(self, frame, feature):
        original = float(pd.to_numeric(frame[feature], errors="coerce").mean()) if feature in frame else float("nan")
        if feature in self.parameters["binary_consensus_features"] or feature == "brand_history":
            used = float(self.raw(frame)[:, self.state["columns"].index(feature)].mean())
        else:
            used = original  # Time remains seconds in the manager-facing observation.
        return used, original


class NeuromaticsModels:
    def __init__(self, frozen, coefficient):
        artifact_hashes()
        self.heads = {"n": NoticeabilityHead(frozen.heads["n"]), "m": frozen.heads["m"], "r": RecallHead()}
        self.coefficient = coefficient
        assert [len(self.heads[t].state["columns"]) for t in "nmr"] == [9, 8, 7]

    def score(self, noticeability, message_delivery, recall, reference_mean=None):
        frames = {t: d.sort_values(["record", "repeat"]).reset_index(drop=True)
                  for t, d in zip("nmr", [noticeability, message_delivery, recall])}
        keys = frames["n"][["record", "repeat"]]
        if keys.empty or keys.duplicated().any() or not all(keys.equals(d[["record", "repeat"]]) for d in frames.values()):
            raise ValueError("Inconsistent prototype repeat keys")
        if any(g.repeat.tolist() != list(range(1, 11)) for _, g in keys.groupby("record")):
            raise ValueError("Exactly ten prototype measurements are required")
        n, m, r = [self.heads[t].predict(frames[t]) for t in "nmr"]
        repeated = keys.assign(noticeability=n, message_delivery=m, norm_ad_recall=r, OPM=n*m, Q=n*m*r)
        out = repeated.groupby("record", as_index=False)[["noticeability", "message_delivery", "norm_ad_recall", "OPM", "Q"]].mean()
        if reference_mean is not None:
            out["C"] = self.coefficient(out.Q, reference_mean)
        return out
