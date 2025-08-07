import pandas as pd

from hrp.data import generate_sample_data
from hrp.scoring import compute_features, score_providers


def test_generate_sample_schema_and_ranges():
    df = generate_sample_data(num_providers=200, seed=123)
    expected_cols = {
        "provider_id",
        "state",
        "num_beneficiaries",
        "num_services",
        "avg_charge",
        "avg_payment",
    }
    assert expected_cols.issubset(set(df.columns))
    assert (df["num_beneficiaries"] > 0).all()
    assert (df["num_services"] > 0).all()
    assert (df["avg_charge"] > 0).all()
    assert (df["avg_payment"] > 0).all()


def test_scoring_outputs_and_ranges():
    df = generate_sample_data(num_providers=500, seed=987)
    enriched = score_providers(df)

    assert "risk_score" in enriched.columns
    assert "risk_label" in enriched.columns

    # Risk score should be within [0, 100]
    assert float(enriched["risk_score"].min()) >= 0.0
    assert float(enriched["risk_score"].max()) <= 100.0

    # Percentile mapping implies there should be providers above 90th percentile when n is large
    assert (enriched["risk_score"] >= 90).sum() >= 1


def test_compute_features_present():
    df = generate_sample_data(num_providers=50, seed=7)
    feats = compute_features(df)
    for c in [
        "services_per_beneficiary",
        "charge_to_payment_ratio",
        "total_charges",
        "total_payments",
    ]:
        assert c in feats.columns