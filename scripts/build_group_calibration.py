"""Print aggregate calibration for signed group effects; no fitting or API calls.

Private historical reference tables remain local. Run from the repository root:
    .venv/bin/python scripts/build_group_calibration.py
The JSON emitted on stdout contains aggregate scales and source hashes only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aipm3.models import (  # noqa: E402
    AIPM1_FEATURES, AIPM1_CAT_FEATURES, AIPM2_FEATURES, AIPM2_CAT_FEATURES,
    AVITO_MEAN_ADRECALL, SCORING_VERSION, load_frozen_models,
)

VERSION = 'frozen-output-iqr-avito-reference-20260908-v1'
HOME_DIR = Path.home()
MODELS_DIR = HOME_DIR / 'Documents/Work projects/AI-Pretest models'
A1_SOURCE = MODELS_DIR / 'v2_decomposition/final_dataset_v2.parquet'
A2_SOURCE = MODELS_DIR / 'v3_adrecall/decomp_v4_gap.parquet'
MD_TRAIN = HOME_DIR / 'outputs/message_delivery_all_features_20260819/train_manifest.csv'


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_digest(frame):
    payload = frame.to_json(orient='split', double_precision=15, index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def dispersion(values):
    """One scale per component, zero anchored; never normalize within a video."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid reference output')
    q10, q25, q50, q75, q90 = np.quantile(values, [.1, .25, .5, .75, .9])
    candidates = [
        ('iqr', float(q75 - q25)),
        ('mad_normal_consistent', float(1.4826 * np.median(np.abs(values - q50)))),
        ('p90_minus_p10', float(q90 - q10)),
        ('full_range_nonrobust_fallback', float(np.ptp(values))),
    ]
    method, scale = next(((m, s) for m, s in candidates if s > 1e-12), (None, None))
    if scale is None:
        raise ValueError('Constant reference output cannot define a meaningful scale')
    return {'scale': scale, 'scale_method': method, 'n': len(values),
            'q25': float(q25), 'median': float(q50), 'q75': float(q75),
            'min': float(values.min()), 'max': float(values.max())}


def aipm1_reference():
    data = pd.read_parquet(A1_SOURCE)
    selected = data.loc[data.OPM_bucket.notna()].copy()
    frame = selected[AIPM1_FEATURES].apply(pd.to_numeric, errors='raise')
    if frame.isna().any().any():
        raise ValueError('Incomplete AIPM1 reference features')
    for column in AIPM1_CAT_FEATURES:
        frame[column] = frame[column].round().astype(int)
    return frame, {'source': 'v2_decomposition/final_dataset_v2.parquet',
                   'source_sha256': file_digest(A1_SOURCE), 'input_rows': len(data),
                   'selection': 'All historical labelled AIPM1 training creatives',
                   'excluded_rows': len(data) - len(selected),
                   'selected_features_sha256': frame_digest(frame)}


def aipm2_reference():
    data = pd.read_parquet(A2_SOURCE)
    valid = data.loc[data['_status'].eq('ok') &
                     data[AIPM2_FEATURES[1:]].notna().all(axis=1)].copy()
    selected = valid.loc[valid.campaign.eq('Avito')].copy()
    if selected.Main_CLIP_ID.duplicated().any():
        raise ValueError('Duplicate Avito creative in AIPM2 reference')

    def frame_for(rows):
        frame = rows[AIPM2_FEATURES[1:]].copy()
        frame['brand_mean_adrecall'] = AVITO_MEAN_ADRECALL
        for column in AIPM2_FEATURES[1:]:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype(int)
        return frame[AIPM2_FEATURES]

    frame = frame_for(selected)
    return frame, frame_for(valid), {
        'source': 'v3_adrecall/decomp_v4_gap.parquet',
        'source_sha256': file_digest(A2_SOURCE), 'input_rows': len(data),
        'valid_rows_all_brands': len(valid),
        'failed_or_incomplete_rows': len(data) - len(valid),
        'other_brand_rows_excluded': len(valid) - len(selected),
        'selection': 'Complete ok Avito creatives only; fixed Avito brand context',
        'brand_mean_adrecall': AVITO_MEAN_ADRECALL,
        'selected_features_sha256': frame_digest(frame),
    }


def md_reference(bundle):
    # These pure readers only merge existing historical data. Their main/search
    # entry points are never invoked, and frozen selection/normalization is reused.
    dependency_site = (HOME_DIR / '.cache/codex-runtimes/codex-primary-runtime/'
                       'dependencies/python/lib/python3.12/site-packages')
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        if not dependency_site.exists():
            raise RuntimeError('Historical source reader requires openpyxl')
        sys.path.append(str(dependency_site))
    sys.path.insert(0, str(HOME_DIR))
    import message_delivery_panel12_model_search as historical

    data, _, _, _ = historical.load_data()
    manifest = pd.read_csv(MD_TRAIN)
    selected = manifest[['row_id']].merge(data, on='row_id', validate='one_to_one')
    if len(selected) != bundle['training_n'] or len(selected) != 48:
        raise ValueError('Unexpected frozen MD training membership')
    for name, expected in bundle['raw_means'].items():
        if not np.isclose(selected[name].mean(), expected, atol=1e-12, rtol=0):
            raise ValueError('MD historical reference differs from frozen normalization: ' + name)
        scale = float(selected[name].std(ddof=0)) or 1.
        if not np.isclose(scale, bundle['raw_scales'][name], atol=1e-12, rtol=0):
            raise ValueError('MD historical reference differs from frozen scale: ' + name)

    def z(name):
        return ((selected[name].astype(float) - bundle['raw_means'][name]) /
                bundle['raw_scales'][name])

    mae = 'p12__cluster_valid_mask_mae_smoothed'
    consistency = 'p12__cluster_valid_mask_consistency'
    specificity = 'message_specificity_level__r3_mean3'
    entity = 'main_idea_entity_load__r4_agreement3'
    audio = 'audio_only_message_completeness__r4_value'
    values = {
        'eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3': abs(z(mae) - z(specificity)),
        'eng__gemini_round4__signed_mean8': sum(z(f) * s for f, s in zip(bundle['round4_top8'], bundle['round4_signs'])) / 8,
        'eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity': z(entity) * z('cta_clarity'),
        'eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3': z(consistency) * z(specificity),
        'eng__product__words_per_second__X__audio_only_message_completeness__r4_value': z('words_per_second') * z(audio),
        'offer_condition_count__r4_mean3': selected['offer_condition_count__r4_mean3'],
        'p12__cluster_valid_mask_mae_smoothed': selected[mae],
        'words_per_second': selected['words_per_second'],
    }
    frame = pd.DataFrame(values)[bundle['feature_columns']]
    if not np.allclose(frame['eng__gemini_round4__signed_mean8'], bundle['signed_mean_train'], atol=1e-12, rtol=0):
        raise ValueError('MD row order or derived features do not match frozen bundle')
    return frame, {
        'source': 'Historical merged MD feature rows; frozen train_manifest.csv',
        'train_manifest_sha256': file_digest(MD_TRAIN), 'input_rows': len(data),
        'selection': 'Exact 48 training rows; excludes fixed 10-video holdout',
        'excluded_rows': len(data) - len(selected),
        'selected_row_ids_sha256': frame_digest(selected[['row_id']]),
        'selected_features_sha256': frame_digest(frame),
        'frozen_raw_normalization_verified': True,
        'frozen_signed_mean_order_verified': True,
    }


def build_calibration():
    models = load_frozen_models(
        str(REPO / 'deployment_artifacts/aipm1_model.cbm'),
        str(REPO / 'deployment_artifacts/aipm2_model.cbm'),
        str(REPO / 'deployment_artifacts/message_delivery_model_bundle.joblib'),
    )
    x1, provenance1 = aipm1_reference()
    x2, x2_all_brands, provenance2 = aipm2_reference()
    xm, provenancem = md_reference(models.message_delivery_bundle)
    logits = np.asarray(models.aipm1.predict(x1, prediction_type='RawFormulaVal'), dtype=float)
    quality1 = logits[:, 2] - logits[:, 0]
    quality2 = models.aipm2.predict(x2)
    qualitym = models.message_delivery_bundle['model'].predict(xm)
    audit_all_brands = dispersion(models.aipm2.predict(x2_all_brands))
    components = {
        'aipm1': {**dispersion(quality1), 'reference_n': len(x1), 'output': 'class2_minus_class0_raw_logit', 'reference': provenance1},
        'aipm2': {**dispersion(quality2), 'reference_n': len(x2), 'output': 'raw_predicted_adrecall', 'reference': provenance2},
        'message_delivery': {**dispersion(qualitym), 'reference_n': len(xm), 'output': 'raw_ordinal_regression_score', 'reference': provenancem},
    }
    return {
        'profile_version': 'creative-group-scores-v2',
        'scoring_version': SCORING_VERSION,
        'calibration_id': VERSION,
        'formula': '50 + 50 * tanh(sum(feature_effects[group]) / component.scale)',
        'neutral': 50,
        'scale_shared_by_all_groups_of_component': True,
        'model_sha256': models.artifact_sha256,
        'components': components,
        'sensitivity_audit': {
            'aipm2_all_valid_brands_fixed_avito_context': audit_all_brands,
            'aipm2_all_brands_to_avito_scale_ratio': audit_all_brands['scale'] / components['aipm2']['scale'],
        },
        'libraries': {name: importlib.metadata.version(name) for name in ['catboost', 'shap', 'scikit-learn', 'numpy']},
        'limitations': [
            'Historical reference cohorts differ by component; scales are frozen, not re-estimated per uploaded video.',
            'Higher means more favorable signed attribution for the corresponding model output, not causal impact or a human response percentage.',
            'AIPM1 explains the high-versus-low logit contrast, not the exact discrete class index or creative_score.',
            'Fifty means zero signed local attribution, not the median creative or an acceptance threshold.',
            'Group values do not sum to the composite AIPM index and cannot be compared as absolute impacts across components.',
            'Calibration does not establish extraction repeatability; validate new independent extraction runs separately.',
        ],
    }


if __name__ == '__main__':
    print(json.dumps(build_calibration(), ensure_ascii=False, indent=2))
