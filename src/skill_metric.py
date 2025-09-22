# src/skill_metrics.py
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------
# Configuration
# ---------------------------

ANNUALIZATION_DAYS = 252  # trading days
FACTOR_COLS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class SummaryMetrics:
    # Sample sizes
    n_obs: int
    n_used: int

    # Return/vol metrics (daily → annualized)
    ann_return: float
    ann_vol: float
    sharpe: float

    # Factor-Model Alpha (annualized) and diagnostics
    alpha_daily: float
    alpha_ann: float
    alpha_t: float
    alpha_p: float
    r2: float

    # Residual-based metrics (skill quality)
    resid_std_daily: float
    tracking_error_ann: float
    alpha_ir: float  # Information Ratio of alpha (annualized alpha / TE)

    # Other useful stats
    hit_ratio: float
    max_drawdown: float
    skew: float
    kurtosis: float


# ---------------------------
# Utilities
# ---------------------------

def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {name} at: {path}")


def _annualize_mean(mean_daily: float, days: int = ANNUALIZATION_DAYS) -> float:
    return mean_daily * days


def _annualize_vol(std_daily: float, days: int = ANNUALIZATION_DAYS) -> float:
    return std_daily * np.sqrt(days)


def _max_drawdown(series: pd.Series) -> float:
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def _infer_df(n: int, k: int) -> int:
    # OLS with intercept: df = n - (k + 1)
    return max(int(n) - (k + 1), 1)


def _alpha_p_value(alpha_t: float, n: int, k: int) -> float:
    df = _infer_df(n, k)
    return float(2.0 * (1.0 - stats.t.cdf(abs(alpha_t), df=df)))


def _to_float(x) -> float:
    return float(np.nan if x is None else x)


# ---------------------------
# Core calculations
# ---------------------------

def load_inputs() -> Tuple[pd.Series, pd.DataFrame, Dict[str, float], Dict[str, float], int, float, float]:
    """Load portfolio returns, factors, and attribution outputs."""
    # Files
    pr_path = DATA / "portfolio_returns.csv"
    ff_path = DATA / "factors.csv"
    at_path = DATA / "portfolio_attribution.csv"

    _require_file(pr_path, "portfolio_returns.csv")
    _require_file(ff_path, "factors.csv")
    _require_file(at_path, "portfolio_attribution.csv")

    # Load
    pf = pd.read_csv(pr_path, parse_dates=["Date"]).set_index("Date").sort_index()
    fac = pd.read_csv(ff_path, parse_dates=["Date"]).set_index("Date").sort_index()
    att = pd.read_csv(at_path)

    if "Portfolio" not in pf.columns:
        raise ValueError("portfolio_returns.csv must contain a 'Portfolio' column.")
    for c in ["RF", *FACTOR_COLS]:
        if c not in fac.columns:
            raise ValueError(f"factors.csv missing column: {c}")

    # Align
    df = pf.join(fac[["RF", *FACTOR_COLS]], how="inner")
    df = df.dropna(subset=["Portfolio", "RF"])
    if df.empty:
        raise ValueError("No overlapping dates between portfolio_returns.csv and factors.csv.")

    # Excess returns
    df["Excess"] = df["Portfolio"] - df["RF"]

    # Read regression outputs
    def _get(col: str) -> float:
        if col not in att.columns:
            raise ValueError(f"portfolio_attribution.csv missing column: {col}")
        return _to_float(att.iloc[0][col])

    alpha = _get("alpha")
    alpha_t = _get("alpha_t")
    r2 = _get("r2")
    n = int(_get("n"))

    betas = {f"beta_{c}": _get(f"beta_{c}") for c in FACTOR_COLS}
    return df["Excess"], df[FACTOR_COLS], betas, {"alpha": alpha, "alpha_t": alpha_t, "r2": r2, "n": n}, n, df.index.min(), df.index.max()


def reconstruct_residuals(excess: pd.Series, X: pd.DataFrame, betas: Dict[str, float], alpha: float) -> pd.Series:
    """epsilon_t = Excess_t - (alpha + sum beta_i * X_i,t)."""
    # Build fitted value using provided betas
    needed = FACTOR_COLS
    if not set(needed).issubset(X.columns):
        missing = set(needed) - set(X.columns)
        raise ValueError(f"X missing factor columns: {missing}")

    beta_vec = pd.Series({c: betas[f"beta_{c}"] for c in FACTOR_COLS}, index=FACTOR_COLS, dtype=float)
    fitted = alpha + (X[FACTOR_COLS] @ beta_vec)
    resid = (excess - fitted).dropna()
    # Align (just in case)
    resid = resid.reindex(excess.index).dropna()
    return resid


def compute_metrics() -> SummaryMetrics:
    excess, X, betas, reg, n_reported, start, end = load_inputs()

    # Daily stats for portfolio (excess series)
    mean_daily = float(excess.mean())
    std_daily = float(excess.std(ddof=1))
    ann_ret = _annualize_mean(mean_daily)
    ann_vol = _annualize_vol(std_daily)
    sharpe = 0.0 if std_daily == 0 else (mean_daily / std_daily) * np.sqrt(ANNUALIZATION_DAYS)

    # Residuals from factor model
    resid = reconstruct_residuals(excess, X, betas, reg["alpha"])
    resid_std = float(resid.std(ddof=1))
    tracking_error_ann = _annualize_vol(resid_std)

    # Alpha/IR
    alpha_daily = float(reg["alpha"])
    alpha_ann = _annualize_mean(alpha_daily)
    alpha_t = float(reg["alpha_t"])
    alpha_p = _alpha_p_value(alpha_t, n=int(reg["n"]), k=len(FACTOR_COLS))

    alpha_ir = 0.0 if tracking_error_ann == 0 else (alpha_ann / tracking_error_ann)

    # Other stats (on portfolio simple returns, not excess)
    # Rebuild simple portfolio returns from excess + RF (aligned on same index)
    # Note: we already aligned in load_inputs; get RF by reloading minimal slice
    rf = pd.read_csv(DATA / "factors.csv", parse_dates=["Date"]).set_index("Date").sort_index()["RF"]
    pf_simple = (excess + rf).reindex(excess.index)
    hit_ratio = float((pf_simple > 0).mean())
    mdd = _max_drawdown(pf_simple)
    skew = float(stats.skew(pf_simple.dropna(), bias=False))
    kurt = float(stats.kurtosis(pf_simple.dropna(), fisher=False, bias=False))

    return SummaryMetrics(
        n_obs=len(excess),
        n_used=len(excess.dropna()),
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        alpha_daily=alpha_daily,
        alpha_ann=alpha_ann,
        alpha_t=alpha_t,
        alpha_p=alpha_p,
        r2=float(reg["r2"]),
        resid_std_daily=resid_std,
        tracking_error_ann=tracking_error_ann,
        alpha_ir=alpha_ir,
        hit_ratio=hit_ratio,
        max_drawdown=mdd,
        skew=skew,
        kurtosis=kurt,
    )


def save_outputs(metrics: SummaryMetrics, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV (rounded to 6 for storage fidelity; display rounding is UI’s job)
    pd.DataFrame([asdict(metrics)]).to_csv(csv_path, index=False)

    # Markdown summary (rounded to 4 as a readable default)
    def r(x): return "nan" if pd.isna(x) else f"{x:.4f}"
    md = [
        "# Portfolio Skill Metrics",
        "",
        f"- Observations (excess): **{metrics.n_obs}**  | Used: **{metrics.n_used}**",
        f"- Period: derived from data in `data/portfolio_returns.csv` / `data/factors.csv`",
        "",
        "## Performance",
        f"- Annualized Return: **{r(metrics.ann_return)}**",
        f"- Annualized Volatility: **{r(metrics.ann_vol)}**",
        f"- Sharpe Ratio: **{r(metrics.sharpe)}**",
        "",
        "## Alpha (Factor Model)",
        f"- Daily Alpha: **{r(metrics.alpha_daily)}**",
        f"- Annualized Alpha: **{r(metrics.alpha_ann)}**",
        f"- Alpha t-stat: **{r(metrics.alpha_t)}**",
        f"- Alpha p-value: **{r(metrics.alpha_p)}**",
        f"- R² (regression): **{r(metrics.r2)}**",
        "",
        "## Residuals (Skill Quality)",
        f"- Residual Std (daily): **{r(metrics.resid_std_daily)}**",
        f"- Tracking Error (annualized): **{r(metrics.tracking_error_ann)}**",
        f"- Alpha Information Ratio: **{r(metrics.alpha_ir)}**",
        "",
        "## Additional",
        f"- Hit Ratio (daily > 0): **{r(metrics.hit_ratio)}**",
        f"- Max Drawdown: **{r(metrics.max_drawdown)}**",
        f"- Skew: **{r(metrics.skew)}**",
        f"- Kurtosis: **{r(metrics.kurtosis)}**",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute portfolio skill metrics (PCAT).")
    parser.add_argument("--data-dir", type=str, default=str(DATA), help="Path to data directory")
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS), help="Path to reports directory")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    logging.info("Starting skill metrics computation")

    metrics = compute_metrics()

    out_csv = data_dir / "skill_metrics.csv"
    out_md = reports_dir / "skill_report.md"
    save_outputs(metrics, out_csv, out_md)

    logging.info("Saved %s", out_csv)
    logging.info("Saved %s", out_md)
    logging.info("Done.")


if __name__ == "__main__":
    main()
