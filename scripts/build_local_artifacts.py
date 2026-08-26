#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import shutil
import zlib
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


FEATURES = [
    "main_character",
    "promo",
    "humor",
    "state_transformation",
    "message_focus_seconds",
    "has_screen_offer_text",
    "unique_offer_count",
]
CAT_FEATURES = ["message_focus_seconds", "has_screen_offer_text"]
MODEL_PARAMS = dict(
    iterations=444,
    depth=5,
    learning_rate=0.05708372532427738,
    l2_leaf_reg=1.5512626640628477,
    border_count=196,
    loss_function="MultiClass",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=False,
)


def encode(path: Path) -> str:
    compressed = zlib.compress(path.read_bytes(), level=9)
    return base64.b64encode(compressed).decode("ascii")


def api_key_from(path: Path) -> str:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    value = data.get("VSELLM_API_KEY") or data.get("general", {}).get("VSELLM_API_KEY")
    if not value:
        raise RuntimeError(f"VSELLM_API_KEY not found in {path}")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen local AIPM 3.0 deployment artifacts")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--aipm2-model", type=Path, required=True)
    parser.add_argument("--md-bundle", type=Path, required=True)
    parser.add_argument("--existing-secrets", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project = args.project_dir.resolve()
    output = project / "deployment_artifacts"
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(args.dataset)
    missing = [column for column in [*FEATURES, "OPM_bucket"] if column not in data.columns]
    if missing:
        raise ValueError(f"Training dataset is missing columns: {missing}")
    train = data.dropna(subset=["OPM_bucket"]).copy()
    frame = train[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    for column in CAT_FEATURES:
        frame[column] = frame[column].round().astype(int)
    target = train["OPM_bucket"].astype(int)

    aipm1 = CatBoostClassifier(**MODEL_PARAMS)
    aipm1.fit(frame, target, cat_features=CAT_FEATURES)
    aipm1_path = output / "aipm1_model.cbm"
    aipm1.save_model(aipm1_path)

    aipm2_path = output / "aipm2_model.cbm"
    md_path = output / "message_delivery_model_bundle.joblib"
    shutil.copy2(args.aipm2_model, aipm2_path)
    shutil.copy2(args.md_bundle, md_path)

    secret_text = (
        f'VSELLM_API_KEY = "{api_key_from(args.existing_secrets)}"\n\n'
        "[artifacts]\n"
        f'"aipm1_model.cbm" = "{encode(aipm1_path)}"\n'
        f'"aipm2_model.cbm" = "{encode(aipm2_path)}"\n'
        f'"message_delivery_model_bundle.joblib" = "{encode(md_path)}"\n'
    )
    local_secrets = project / ".streamlit" / "secrets.toml"
    cloud_secrets = project / "SECRETS_FOR_STREAMLIT.toml"
    local_secrets.write_text(secret_text, encoding="utf-8")
    cloud_secrets.write_text(secret_text, encoding="utf-8")

    accuracy = float((aipm1.predict(frame).reshape(-1).astype(int) == target.to_numpy()).mean())
    print(f"AIPM1 frozen: n={len(train)}, train accuracy={accuracy:.3f}")
    for path in [aipm1_path, aipm2_path, md_path, local_secrets, cloud_secrets]:
        print(f"Created {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
