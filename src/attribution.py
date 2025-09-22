# src/attribution.py
from __future__ import annotations

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------- Config ----------
XCOLS: List[str] = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
HAC_LAGS: int = 5  # Newey–West lags for robust SEs


# ---------- Core helpers ----------
def _req(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {name} at: {path}")


def _load_for_factor_regression():
    p = DATA / "portfolio_returns.csv"
    f = DATA / "factors.csv"
    _req(p, "portfolio_returns.csv")
    _req(f, "factors.csv")

    pf = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    fac = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()

    required = ["RF", *XCOLS]
    missing = [c for c in required if c not in fac.columns]
    if missing:
        raise ValueError(f"factors.csv missing columns: {missing}")

    # align and compute excess returns
    df = pf.join(fac[["RF", *XCOLS]], how="inner").dropna(subset=["Portfolio", "RF"])
    if df.empty:
        raise ValueError("No overlapping dates between portfolio_returns.csv and factors.csv.")
    df["Excess"] = df["Portfolio"] - df["RF"]
    return df


def _fit_hac_ols(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


# ---------- 6-factor attribution (your original logic, wrapped) ----------
def run_factor_attribution() -> pd.DataFrame:
    df = _load_for_factor_regression()
    X = df[XCOLS]
    y = df["Excess"]

    res = _fit_hac_ols(y, X, lags=HAC_LAGS)

    out = {
        "alpha": res.params.get("const", np.nan),
        "alpha_t": res.tvalues.get("const", np.nan),
        "r2": res.rsquared,
        "n": int(res.nobs),
        **{f"beta_{c}": res.params.get(c, np.nan) for c in XCOLS},
        **{f"t_{c}": res.tvalues.get(c, np.nan) for c in XCOLS},
    }
    df_out = pd.DataFrame([out])
    df_out.to_csv(DATA / "portfolio_attribution.csv", index=False)
    print("Saved", DATA / "portfolio_attribution.csv")
    print(df_out.T.round(6))
    return df_out


# ---------- Timing skill tests (Treynor–Mazuy & Henriksson–Merton) ----------
def _load_for_timing():
    p = DATA / "portfolio_returns.csv"
    f = DATA / "factors.csv"
    _req(p, "portfolio_returns.csv")
    _req(f, "factors.csv")

    pf = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    fac = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()

    for c in ["RF", "MKT"]:
        if c not in fac.columns:
            raise ValueError(f"factors.csv missing column: {c}")

    df = pf.join(fac[["RF", "MKT"]], how="inner").dropna(subset=["Portfolio", "RF", "MKT"])
    if df.empty:
        raise ValueError("No overlapping dates between portfolio_returns.csv and factors.csv.")
    df["Excess"] = df["Portfolio"] - df["RF"]
    return df


def run_timing_tests() -> pd.DataFrame:
    df = _load_for_timing()

    # Treynor–Mazuy: Excess_t = a + b*MKT_t + g*MKT_t^2 + e_t
    X_tm = pd.DataFrame({"MKT": df["MKT"], "MKT2": df["MKT"] ** 2}, index=df.index)
    res_tm = _fit_hac_ols(df["Excess"], X_tm, lags=HAC_LAGS)
    row_tm = {
        "model": "Treynor-Mazuy",
        "alpha": res_tm.params.get("const", np.nan),
        "t_alpha": res_tm.tvalues.get("const", np.nan),
        "beta_MKT": res_tm.params.get("MKT", np.nan),
        "t_MKT": res_tm.tvalues.get("MKT", np.nan),
        "gamma_MKT2": res_tm.params.get("MKT2", np.nan),
        "t_MKT2": res_tm.tvalues.get("MKT2", np.nan),
        "r2": res_tm.rsquared,
        "n": int(res_tm.nobs),
    }

    # Henriksson–Merton: Excess_t = a + b_d*D_t*MKT_t + b_u*(1-D_t)*MKT_t + e_t
    D = (df["MKT"] < 0).astype(int)
    X_hm = pd.DataFrame(
        {"MKT_down": D * df["MKT"], "MKT_up": (1 - D) * df["MKT"]},
        index=df.index,
    )
    res_hm = _fit_hac_ols(df["Excess"], X_hm, lags=HAC_LAGS)
    row_hm = {
        "model": "Henriksson-Merton",
        "alpha": res_hm.params.get("const", np.nan),
        "t_alpha": res_hm.tvalues.get("const", np.nan),
        "beta_down": res_hm.params.get("MKT_down", np.nan),
        "t_beta_down": res_hm.tvalues.get("MKT_down", np.nan),
        "beta_up": res_hm.params.get("MKT_up", np.nan),
        "t_beta_up": res_hm.tvalues.get("MKT_up", np.nan),
        "timing_spread": (res_hm.params.get("MKT_up", np.nan) - res_hm.params.get("MKT_down", np.nan)),
        "r2": res_hm.rsquared,
        "n": int(res_hm.nobs),
    }

    out = pd.DataFrame([row_tm, row_hm])
    out.to_csv(DATA / "timing_tests.csv", index=False)
    print("Saved", DATA / "timing_tests.csv")
    print(out.round(6).T)
    return out


# ---------- Orchestration ----------
def main():
    run_factor_attribution()  # existing 6-factor regression
    try:
        run_timing_tests()     # timing skill (TM & HM)
    except Exception as e:
        print("[WARN] Timing tests failed:", e)


if __name__ == "__main__":
    main()
