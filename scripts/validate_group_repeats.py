"""Offline validation of original group scores; never tunes their scale.

Inputs are private complete result JSONs. Publish aggregate checks only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aipm3.feature_profile import build_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    details, groups, sources = [], {}, []
    for video in [1, 2, 3]:
        files = [args.baseline_dir / f"video_{video}.json"] + [
            args.repeat_dir / f"video_{video}_repeat_{run}.json" for run in [1, 2]]
        results = [json.loads(path.read_text()) for path in files]
        assert len({r["source_sha"] for r in results}) == 1
        assert len({r["video_sha"] for r in results}) == 1
        profiles = []
        for path, result in zip(files, results):
            original = copy.deepcopy(result)
            profiles.append(build_profile(result))
            assert result == original
            assert build_profile(result) == profiles[-1]
            sources.append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        assert all(len(p["groups"]) == 11 for p in profiles)
        for index, row in enumerate(profiles[0]["groups"]):
            values = [p["groups"][index]["score"] for p in profiles]
            assert all(v is not None for v in values)
            delta = max(values) - min(values)
            stats = groups.setdefault(row["label"], {"component": row["component"], "max_range": 0.0})
            stats["max_range"] = max(stats["max_range"], delta)
            details.append({"video": video, "group": row["label"], "scores": values,
                            "range": delta, "within_10_points": delta <= 10})
    unstable = [group for group, stats in groups.items() if stats["max_range"] > 10]
    public = {"completed": True, "videos": 3, "fresh_repeats_per_video": 2,
              "comparison_runs_per_video": 3, "engineering_tolerance_points": 10,
              "groups": groups, "unstable_groups": unstable,
              "pass_group_video_pairs": sum(r["within_10_points"] for r in details),
              "group_video_pairs": len(details),
              "summary": "Пилот: 3 ролика, исходный запуск и по 2 полностью свежих разметки. "
              "Проверены все входы: AIPM1, AIPM2, панель 30, recovery и транскрипция. "
              f"Размах не более 10 баллов: {sum(r['within_10_points'] for r in details)} из {len(details)} "
              "пар «ролик — группа». Это инженерный допуск, не доказательство эквивалентности. "
              "Три ролика не гарантируют такую же устойчивость на других креативах."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"public": public, "details": details,
                                     "source_fingerprints": sources}, ensure_ascii=False, indent=2))
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
