# src/forecast.py
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    # Optional (for VAR); code falls back gracefully if unavailable
    from statsmodels.tsa.api import VAR
    _HAS_VAR = True
except Exception:
    _HAS_VAR = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FACTOR_COLS: List[str] = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
ANNUALIZATION_DAYS = 252


# ---------------------------
# Utilities / IO
# ---------------------------
def _req(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {name} at: {path}")


def load_inputs() -> Tuple[pd.Series, pd.DataFrame, Dict[str, float], float]:
    """
    Load:
      - portfolio simple returns (Series 'Portfolio')
      - factors dataframe with RF and factor columns
      - betas dict (beta_c) and alpha from attribution.csv
    Returns (pf_simple, factors, betas, alpha)
    """
    pr = DATA / "portfolio_returns.csv"
    ff = DATA / "factors.csv"
    at = DATA / "portfolio_attribution.csv"

    _req(pr, "portfolio_returns.csv")
    _req(ff, "factors.csv")
    _req(at, "portfolio_attribution.csv")

    pf = pd.read_csv(pr, parse_dates=["Date"]).set_index("Date").sort_index()["Portfolio"]
    fac = pd.read_csv(ff, parse_dates=["Date"]).set_index("Date").sort_index()

    for c in ["RF", *FACTOR_COLS]:
        if c not in fac.columns:
            raise ValueError(f"factors.csv missing column: {c}")

    att = pd.read_csv(at)
    alpha = float(att.iloc[0]["alpha"])
    betas = {f"beta_{c}": float(att.iloc[0][f"beta_{c}"]) for c in FACTOR_COLS}

    # Make sure aligned
    common = pf.index.intersection(fac.index)
    pf = pf.loc[common]
    fac = fac.loc[common]

    return pf, fac, betas, alpha


def reconstruct_residuals(excess: pd.Series, X: pd.DataFrame, betas: Dict[str, float], alpha: float) -> pd.Series:
    beta_vec = pd.Series({c: betas[f"beta_{c}"] for c in FACTOR_COLS}, index=FACTOR_COLS, dtype=float)
    fitted = alpha + (X[FACTOR_COLS] @ beta_vec)
    resid = (excess - fitted).dropna()
    return resid.reindex(excess.index).dropna()


# ---------------------------
# Forecast models for factors
# ---------------------------
def forecast_factors_mean(factors: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Historical mean (per-day) forecast for each factor → horizon-day mean equals 1-day mean.
    """
    mu = factors[FACTOR_COLS].mean()
    return mu


def forecast_factors_ewma(factors: pd.DataFrame, horizon: int, lam: float = 0.94) -> pd.Series:
    """
    EWMA (exponentially weighted mean). lam ~ 0.94 is common (RiskMetrics style).
    """
    w = (1 - lam) * np.power(lam, np.arange(len(factors))[::-1])
    w = w / w.sum()
    W = np.tile(w.reshape(-1, 1), (1, len(FACTOR_COLS)))
    x = factors[FACTOR_COLS].values
    mu = pd.Series((W * x).sum(axis=0), index=FACTOR_COLS)
    return mu


def forecast_factors_var1(factors: pd.DataFrame, horizon: int, maxlags: int = 1) -> pd.Series:
    """
    VAR(1) one-step forecast; for horizon>1 we repeatedly iterate.
    Falls back to historical mean if statsmodels VAR is not available.
    """
    if not _HAS_VAR:
        return forecast_factors_mean(factors, horizon)
    model = VAR(factors[FACTOR_COLS].dropna())
    res = model.fit(maxlags)
    y = factors[FACTOR_COLS].iloc[-1].values
    # iterate horizon steps
    for _ in range(horizon):
        # y_{t+1} = c + A y_t (VAR(1) implied step)
        # statsmodels provides .forecast with a window; we'll just use it for multi-step
        y = res.forecast(y.reshape(1, -1), steps=1)[0]
    return pd.Series(y, index=FACTOR_COLS)


# ---------------------------
# Projection logic
# ---------------------------
@dataclass
class ForecastResult:
    horizon_days: int
    model: str
    alpha_daily: float
    expected_excess_daily: float
    expected_excess_annual: float
    ci95_daily_low: float
    ci95_daily_high: float


def compute_expected_excess(mu_factors_daily: pd.Series, betas: Dict[str, float], alpha: float) -> float:
    beta_vec = pd.Series({c: betas[f"beta_{c}"] for c in FACTOR_COLS}, index=FACTOR_COLS, dtype=float)
    return float(alpha + float(mu_factors_daily @ beta_vec))


def daily_to_annual(mu_daily: float) -> float:
    # Good enough linearization for small daily returns; for compounding use (1+mu)**252-1
    return mu_daily * ANNUALIZATION_DAYS


def estimate_resid_sigma(excess: pd.Series, X: pd.DataFrame, betas: Dict[str, float], alpha: float) -> float:
    resid = reconstruct_residuals(excess, X, betas, alpha)
    return float(resid.std(ddof=1))


def run_forecast(horizon: int, model: str, ewma_lambda: float) -> Tuple[ForecastResult, pd.DataFrame]:
    """
    Returns:
      - ForecastResult summary
      - scenarios DataFrame with +/-1σ shocks per factor (impact on excess)
    """
    pf, fac, betas, alpha = load_inputs()
    excess = (pf - fac["RF"]).dropna()
    X = fac[FACTOR_COLS].reindex(excess.index)

    # choose factor forecast
    if model == "mean":
        mu = forecast_factors_mean(X, horizon)
    elif model == "ewma":
        mu = forecast_factors_ewma(X, horizon, lam=ewma_lambda)
    elif model == "var1":
        mu = forecast_factors_var1(X, horizon, maxlags=1)
    else:
        raise ValueError("model must be one of: mean, ewma, var1")

    # expected daily excess return
    mu_ex_daily = compute_expected_excess(mu, betas, alpha)
    mu_ex_ann = daily_to_annual(mu_ex_daily)

    # uncertainty proxy from residuals (idiosyncratic noise of the factor model)
    sigma_resid = estimate_resid_sigma(excess, X, betas, alpha)
    # One-day 95% CI (approx normal); for H>1, widen by sqrt(H)
    ci_scale = np.sqrt(max(horizon, 1))
    ci_low = mu_ex_daily - 1.96 * sigma_resid / ci_scale
    ci_high = mu_ex_daily + 1.96 * sigma_resid / ci_scale

    summary = ForecastResult(
        horizon_days=horizon,
        model=model,
        alpha_daily=alpha,
        expected_excess_daily=mu_ex_daily,
        expected_excess_annual=mu_ex_ann,
        ci95_daily_low=ci_low,
        ci95_daily_high=ci_high,
    )

    # Scenario table: +/- 1σ factor shocks (using recent std dev)
    fac_sigma = X[FACTOR_COLS].std(ddof=1)
    beta_vec = pd.Series({c: betas[f"beta_{c}"] for c in FACTOR_COLS}, index=FACTOR_COLS, dtype=float)

    scen = []
    base = mu_ex_daily
    for c in FACTOR_COLS:
        shock = float(fac_sigma[c])
        # Only shock the target factor; keep others unchanged
        mu_up = mu.add(pd.Series({c: shock}),    fill_value=0.0)
        mu_dn = mu.add(pd.Series({c: -shock}),   fill_value=0.0)
        up = float(alpha + float(mu_up @ beta_vec))
        dn = float(alpha + float(mu_dn @ beta_vec))
        scen.append({
            "factor": c,
            "sigma": shock,
            "impact_up": up - base,
            "impact_down": dn - base
        })
    scenarios = pd.DataFrame(scen)

    return summary, scenarios


def save_outputs(summary: ForecastResult, scenarios: pd.DataFrame, model_tag: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_csv = DATA / f"forecast_{model_tag}.csv"
    scen_csv = DATA / f"forecast_scenarios_{model_tag}.csv"
    md_path = REPORTS / f"forecast_report_{model_tag}.md"

    # Save CSVs
    pd.DataFrame([asdict(summary)]).to_csv(out_csv, index=False)
    scenarios.to_csv(scen_csv, index=False)

    # Markdown report
    def r(x): return "nan" if pd.isna(x) else f"{x:.6f}"
    md = [
        f"# Forecast Report ({model_tag})",
        "",
        f"- Horizon (days): **{summary.horizon_days}**",
        f"- Model: **{summary.model.upper()}**",
        f"- Daily alpha (from attribution): **{r(summary.alpha_daily)}**",
        "",
        "## Expected Excess Return",
        f"- Daily: **" + r(summary.expected_excess_daily) + "**",
        f"- Annualized (×252): **" + r(summary.expected_excess_annual) + "**",
        "",
        "## Uncertainty (95% CI, daily)",
        f"- Lower: **" + r(summary.ci95_daily_low) + "**",
        f"- Upper: **" + r(summary.ci95_daily_high) + "**",
        "",
        "## Factor Shock Scenarios (±1σ, daily impact on excess)",
        scenarios.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("Saved:", out_csv)
    print("Saved:", scen_csv)
    print("Saved:", md_path)


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Factor-based portfolio return forecast (PCAT).")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in trading days")
    parser.add_argument("--model", type=str, default="ewma", choices=["mean", "ewma", "var1"], help="Factor forecast model")
    parser.add_argument("--ewma-lambda", type=float, default=0.94, help="EWMA decay (if model=ewma)")
    args = parser.parse_args()

    summ, scen = run_forecast(horizon=args.horizon, model=args.model, ewma_lambda=args.ewma_lambda)
    save_outputs(summ, scen, model_tag=f"{args.model}_{args.horizon}d")


if __name__ == "__main__":
    main()
