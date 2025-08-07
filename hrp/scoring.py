from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


_FEATURE_COLUMNS: List[str] = [
    "services_per_beneficiary",
    "charge_to_payment_ratio",
    "total_charges",
    "total_payments",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # Guard against zero beneficiaries
    denom_benes = np.maximum(result.get("num_beneficiaries", 1), 1)
    denom_payment = np.maximum(result.get("avg_payment", 0.0).astype(float), 1e-6)

    result["services_per_beneficiary"] = (
        result.get("num_services", 0).astype(float) / denom_benes.astype(float)
    )
    result["charge_to_payment_ratio"] = (
        result.get("avg_charge", 0).astype(float) / denom_payment
    )
    result["total_charges"] = result.get("avg_charge", 0).astype(float) * result.get(
        "num_services", 0
    ).astype(float)
    result["total_payments"] = result.get("avg_payment", 0).astype(float) * result.get(
        "num_services", 0
    ).astype(float)

    return result


def _positive_zscores(values: pd.Series) -> pd.Series:
    vals = values.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = vals.mean()
    std = vals.std(ddof=0)
    if std <= 1e-12:
        z = pd.Series(0.0, index=vals.index)
    else:
        z = (vals - mean) / std
    return z.clip(lower=0.0)


def score_providers(df: pd.DataFrame) -> pd.DataFrame:
    enriched = compute_features(df)

    # Compute positive-side z-scores for features where higher implies potential risk
    pos_z = [
        _positive_zscores(enriched[col]) for col in _FEATURE_COLUMNS if col in enriched
    ]

    if not pos_z:
        enriched["risk_raw"] = 0.0
    else:
        z_matrix = pd.concat(pos_z, axis=1)
        z_matrix.columns = [col for col in _FEATURE_COLUMNS if col in enriched]
        enriched["risk_raw"] = z_matrix.mean(axis=1)

    # Convert to percentile-based 0-100 score for interpretability
    # If all risk_raw are equal, assign zeros
    if enriched["risk_raw"].nunique(dropna=False) <= 1:
        enriched["risk_score"] = 0.0
    else:
        enriched["risk_score"] = enriched["risk_raw"].rank(pct=True) * 100.0

    # Discrete labels for quick triage
    def label_row(score: float) -> str:
        if score >= 90:
            return "high"
        if score >= 70:
            return "elevated"
        if score >= 40:
            return "medium"
        return "low"

    enriched["risk_label"] = enriched["risk_score"].apply(label_row)

    return enriched