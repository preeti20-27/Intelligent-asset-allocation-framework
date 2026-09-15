"""Sanity-check allocator.hrp.allocate() against PyPortfolioOpt's HRPOpt.

Generates a synthetic multi-asset returns DataFrame, runs both:
  - PyPortfolioOpt's built-in HRPOpt (single-linkage HRP reference impl), and
  - this project's own allocate() from allocator/hrp.py,

then prints both weight vectors side by side along with the per-asset
absolute difference, so the two can be visually compared.

Usage:
    python scripts/validate_against_reference.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as `python scripts/validate_against_reference.py` from
# anywhere without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allocator.hrp import allocate  # noqa: E402

try:
    from pypfopt import HRPOpt
except ImportError:
    print(
        "PyPortfolioOpt is not installed. Install it with:\n"
        "    pip install PyPortfolioOpt\n"
    )
    raise


def generate_sample_returns(n_assets=10, n_days=750, seed=42) -> pd.DataFrame:
    """Synthetic multi-asset daily returns with a factor-model correlation structure."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    n_factors = 4
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio_std = rng.uniform(0.003, 0.03, size=n_assets)  # varied volatilities
    idio = rng.normal(0, 1, size=(n_days, n_assets)) * idio_std

    returns = factors @ factor_loadings.T + idio
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


def main():
    returns_df = generate_sample_returns()

    # --- Reference: PyPortfolioOpt HRPOpt ---
    hrp_ref = HRPOpt(returns=returns_df)
    ref_weights_dict = hrp_ref.optimize()
    ref_weights = pd.Series(ref_weights_dict).sort_index()

    # --- Ours: allocator.hrp.allocate ---
    our_weights = allocate(returns_df).sort_index()

    comparison = pd.DataFrame(
        {
            "pypfopt_HRP": ref_weights,
            "allocator_hrp": our_weights,
        }
    )
    comparison["abs_diff"] = (comparison["pypfopt_HRP"] - comparison["allocator_hrp"]).abs()

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("=" * 60)
    print("HRP weight comparison: PyPortfolioOpt vs allocator.hrp")
    print("=" * 60)
    print(comparison.to_string())
    print("-" * 60)
    print(f"Max absolute difference across assets: {comparison['abs_diff'].max():.6f}")
    print(f"Mean absolute difference across assets: {comparison['abs_diff'].mean():.6f}")
    print(f"Sum of weights - pypfopt:  {comparison['pypfopt_HRP'].sum():.6f}")
    print(f"Sum of weights - allocator: {comparison['allocator_hrp'].sum():.6f}")


if __name__ == "__main__":
    main()
