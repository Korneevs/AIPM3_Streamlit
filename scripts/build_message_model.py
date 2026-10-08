"""Rebuild the eight-input message head and its existing display references.

No model selection, threshold search, video extraction or API calls. The
October 1 and October 2 bundles remain immutable rollback points.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from aipm3 import latest_runtime as runtime
from aipm3.neuromatics_models import NeuromaticsModels

REMOVED = "numeric_offer_on_screen"
OUTPUT = APP / "aipm3/message_bundle/20261008"
FINISHED_VERSION = "three-heads-no-screen-number-20261008"
NEURO_VERSION = "neuromatics-no-screen-number-20261008"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metrics(frame, column):
    return dict(spearman=float(spearmanr(frame.y, frame[column]).statistic),
                pearson=float(pearsonr(frame.y, frame[column]).statistic),
                mae=float(np.mean(abs(frame.y - frame[column]))),
                rmse=float(np.mean((frame.y - frame[column]) ** 2) ** .5))


def family_scores(model, frames):
    scores = model.score(*(frames[t] for t in "nmr"))
    scores = scores.merge(frames["n"][["record", "family"]].drop_duplicates(),
                          on="record", validate="one_to_one")
    return scores.groupby("family").Q.mean().sort_index()


def main(root):
    module = runtime.model_module()
    before_hashes = runtime.artifact_hashes()
    base = module.AIPM3(runtime.BUNDLE_DIR / "models")
    spec = deepcopy(base.heads["m"].state["spec"])
    original_spec = deepcopy(spec)
    spec["fixed"].remove(REMOVED)
    train = pd.read_csv(runtime.BUNDLE_DIR / "data/fit_m.csv", float_precision="round_trip")
    loo = []
    for family in sorted(train.family.unique()):
        fitting, held = train.loc[train.family != family], train.loc[train.family == family]
        row = held[["record", "family", "sha", "y"]].copy()
        row["prediction"] = module.Head.fit(fitting, deepcopy(spec)).predict(held)
        row["previous"] = module.Head.fit(fitting, deepcopy(original_spec)).predict(held)
        loo.append(row.groupby(["record", "family", "sha"], as_index=False).agg(
            y=("y", "first"), prediction=("prediction", "mean"), previous=("previous", "mean")))
    loo = pd.concat(loo, ignore_index=True)
    baseline = pd.read_csv(runtime.BUNDLE_DIR / "human_loo_m.csv", float_precision="round_trip")
    check = loo.merge(baseline, on="record", suffixes=("", "_saved"), validate="one_to_one")
    np.testing.assert_allclose(check.previous, check.prediction_saved, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(check.y, check.y_saved)
    model = module.AIPM3(runtime.BUNDLE_DIR / "models")
    model.heads["m"] = module.Head.fit(train, spec)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.heads["m"].save(OUTPUT / "models/m")
    loo.to_csv(OUTPUT / "human_loo_m.csv", index=False)
    metadata = dict(scoring_version=FINISHED_VERSION, neuromatics_scoring_version=NEURO_VERSION,
                    removed_feature=REMOVED, features={"n": 9, "m": 8, "r": 7},
                    validation="family-LOO; fixed settings, no feature or parameter search",
                    records=len(loo), families=int(loo.family.nunique()),
                    previous=metrics(loo, "previous"), current=metrics(loo, "prediction"),
                    dtb_ev_recomputed=False, business_effect_recomputed=False)
    dump(OUTPUT / "metrics.json", metadata)

    # Preserve the exact historical campaign pool and equal family weights.
    campaign_dir = root / "outputs/AIPM_effect15_20261001/recommended/data"
    paths = {t: campaign_dir / f"campaign_inputs_{t}.csv" for t in "nmr"}
    frames = {t: pd.read_csv(path, float_precision="round_trip") for t, path in paths.items()}
    current_ref = json.loads((APP / "aipm3/latest_display_reference.json").read_text())
    old_q = family_scores(base, frames)
    np.testing.assert_allclose(old_q, pd.Series(current_ref["family_Q"]).reindex(old_q.index),
                               rtol=0, atol=1e-12)
    q = family_scores(model, frames)
    assert set(q.index) == set(current_ref["family_Q"])
    assert len(frames["n"].record.unique()) == current_ref["campaigns"]
    ref = dict(current_ref, scoring_version=FINISHED_VERSION,
               Q_mean=float(q.mean()), Q_tertiles=np.quantile(q, [1 / 3, 2 / 3]).tolist(),
               family_Q=q.to_dict(),
               source_hashes={t: hashlib.sha256(path.read_bytes()).hexdigest() for t, path in paths.items()})
    dump(OUTPUT / "finished_display_reference.json", ref)

    # Use the same twelve pairs and the existing median-log-ratio mapping.
    pairs_path = root / "outputs/AIPM_pairs_expansion_20261002/primary_pairs.json"
    pairs = json.loads(pairs_path.read_text())
    pair_frames = {t: pd.read_csv(root / f"outputs/AIPM_neuromatics_expanded_20261002/canonical_{t}.csv",
                                 float_precision="round_trip") for t in "nmr"}
    neuro = NeuromaticsModels(model, module.coefficient)
    rows = []
    for pair in pairs:
        row = dict(pair_id=pair["pair_id"])
        for role, estimator, sha in [("prototype", neuro, pair["prototype"]["video_sha256"]),
                                     ("finished", model, pair["finished_sha256"])]:
            predictions = {}
            for task in "nmr":
                f = pair_frames[task].loc[pair_frames[task].sha == sha].sort_values("repeat")
                assert f.repeat.tolist() == list(range(1, 11))
                predictions[task] = estimator.heads[task].predict(f)
                row[role + "_" + task] = float(predictions[task].mean())
            row[role + "_Q"] = float((predictions["n"] * predictions["m"] * predictions["r"]).mean())
        rows.append(row)
    pair_scores = pd.DataFrame(rows)
    pair_scores.to_csv(OUTPUT / "display_pair_scores.csv", index=False)
    calibration = json.loads((APP / "aipm3/neuromatics_display_reference.json").read_text())
    assert len(pairs) == calibration["pair_count"] == 12
    # Noticeability and recall and their reference definitions are untouched.
    for task in ["m", "Q"]:
        if task == "m":
            bg = train.sort_values(["family", "record", "repeat"]).groupby("family", sort=True).nth(0)
            values = model.heads[task].predict(bg)
            finished = dict(mean=float(values.mean()), cuts=np.quantile(values, [1 / 3, 2 / 3]).tolist())
        else:
            finished = dict(mean=ref["Q_mean"], cuts=ref["Q_tertiles"])
        factor = float(np.exp(np.median(np.log(pair_scores["finished_" + task] / pair_scores["prototype_" + task]))))
        calibration["references"][task] = dict(mean=finished["mean"] / factor,
            cuts=[v / factor for v in finished["cuts"]], finished_scale_factor=factor, finished_reference=finished)
    calibration.update(version="neuromatics-display-pairs12-median-log-m8-20261008",
                       scoring_version=NEURO_VERSION,
                       validation="fixed_mapping_rebuilt_on_same_pairs; new_class_validation_not_run")
    dump(OUTPUT / "neuromatics_display_reference.json", calibration)
    dump(OUTPUT / "provenance.json", dict(base_manifest=before_hashes,
         training_sha256=before_hashes["data/fit_m.csv"],
         pairs_sha256=hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
         pair_input_sha256={t: hashlib.sha256((root / f"outputs/AIPM_neuromatics_expanded_20261002/canonical_{t}.csv").read_bytes()).hexdigest() for t in "nmr"},
         changed_head="m", retained_settings=spec, upstream_extraction_unchanged=True))
    files = [path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "manifest.json"]
    dump(OUTPUT / "manifest.json", {str(path.relative_to(OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)})
    assert runtime.artifact_hashes() == before_hashes
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    main(parser.parse_args().research_root.resolve())
