#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import sys
import zlib
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aipm3.models import EXPECTED_ARTIFACT_SHA256


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
    parser = argparse.ArgumentParser(description="Package pinned AIPM 3.0 artifacts without retraining")
    parser.add_argument("--aipm1-model", type=Path, required=True)
    parser.add_argument("--aipm2-model", type=Path, required=True)
    parser.add_argument("--md-bundle", type=Path, required=True)
    parser.add_argument("--existing-secrets", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    sources = {"aipm1": args.aipm1_model, "aipm2": args.aipm2_model, "message_delivery": args.md_bundle}
    for name, source in sources.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_ARTIFACT_SHA256[name]:
            raise ValueError(f"{name}: source does not match the pinned artifact; nothing was written")
    api_key = api_key_from(args.existing_secrets)
    project = args.project_dir.resolve()
    output = project / "deployment_artifacts"
    output.mkdir(parents=True, exist_ok=True)
    aipm1_path = output / "aipm1_model.cbm"
    aipm2_path = output / "aipm2_model.cbm"
    md_path = output / "message_delivery_model_bundle.joblib"
    for source, target in [(args.aipm1_model, aipm1_path), (args.aipm2_model, aipm2_path), (args.md_bundle, md_path)]:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    secret_text = (
        f'VSELLM_API_KEY = "{api_key}"\n\n'
        "[artifacts]\n"
        f'"aipm1_model.cbm" = "{encode(aipm1_path)}"\n'
        f'"aipm2_model.cbm" = "{encode(aipm2_path)}"\n'
        f'"message_delivery_model_bundle.joblib" = "{encode(md_path)}"\n'
    )
    local_secrets = project / ".streamlit" / "secrets.toml"
    cloud_secrets = project / "SECRETS_FOR_STREAMLIT.toml"
    local_secrets.parent.mkdir(parents=True, exist_ok=True)
    local_secrets.write_text(secret_text, encoding="utf-8")
    cloud_secrets.write_text(secret_text, encoding="utf-8")

    print("All three pinned artifacts verified; no training performed")
    for path in [aipm1_path, aipm2_path, md_path, local_secrets, cloud_secrets]:
        print(f"Created {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
