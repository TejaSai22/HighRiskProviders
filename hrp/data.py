from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]


def generate_sample_data(num_providers: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic provider dataset with plausible distributions.

    Columns:
    - provider_id: unique identifier
    - state: US state
    - num_beneficiaries: positive integer
    - num_services: positive integer, correlated with beneficiaries
    - avg_charge: average submitted charge amount per service
    - avg_payment: average medicare payment amount per service
    """
    rng = np.random.default_rng(seed)

    provider_id = np.arange(1, num_providers + 1)
    state = rng.choice(_US_STATES, size=num_providers)

    # Beneficiaries and services with over-dispersion
    num_beneficiaries = rng.negative_binomial(n=20, p=0.5, size=num_providers) + 1
    base_services = num_beneficiaries * rng.uniform(1.2, 3.5, size=num_providers)
    num_services = np.maximum(1, np.round(base_services + rng.normal(0, 5, size=num_providers))).astype(int)

    # Charges and payments
    avg_charge = rng.gamma(shape=2.5, scale=80.0, size=num_providers) + 20
    # Payments are a proportion of charges with some noise, capped to be <= charge
    pay_ratio = np.clip(rng.normal(0.55, 0.08, size=num_providers), 0.2, 0.95)
    avg_payment = np.minimum(avg_charge * pay_ratio, avg_charge * 0.98)

    df = pd.DataFrame(
        {
            "provider_id": provider_id,
            "state": state,
            "num_beneficiaries": num_beneficiaries,
            "num_services": num_services,
            "avg_charge": avg_charge,
            "avg_payment": avg_payment,
        }
    )

    return df


def _is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def load_providers(input_path: Optional[str]) -> pd.DataFrame:
    """Load provider dataset from CSV.

    If the path is None, does not exist, or is a Git LFS pointer, returns synthetic data.
    """
    if not input_path:
        return generate_sample_data()

    path = Path(input_path)
    if not path.exists() or path.is_dir():
        print(f"[hrp] Input path '{path}' not found. Using synthetic data.")
        return generate_sample_data()

    # Detect LFS placeholder
    if _is_git_lfs_pointer(path):
        print(f"[hrp] Detected Git LFS pointer at '{path}'. Using synthetic data.")
        return generate_sample_data()

    try:
        df = pd.read_csv(path, low_memory=False)
        # Basic sanity check for expected numeric columns; if missing, bail to synthetic
        required_like = {"num_services", "avg_charge", "avg_payment"}
        if not required_like.issubset(set(df.columns)):
            print(
                f"[hrp] Input at '{path}' missing expected columns {required_like}. Using synthetic data."
            )
            return generate_sample_data()
        return df
    except Exception as exc:
        print(f"[hrp] Failed to read '{path}' ({exc}). Using synthetic data.")
        return generate_sample_data()


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)