# src/visuals.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype as _is_dt

# Try to ensure Kaleido can find Chrome for PNG export (no-op if already set)
try:
    import kaleido  # type: ignore
    kaleido.get_chrome_sync()
except Exception:
    pass

# ---------------------------
# Paths & Config
# ---------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

FACTOR_COLS: List[str] = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
ROLL_WINDOW_DEFAULT = 126  # ~6 months

PLOTLY_TEMPLATE = "plotly_white"
COLORWAY = px.colors.qualitative.D3


# ---------------------------
# Helpers
# ---------------------------
def _req(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {name} at: {path}")


def _normalize_index_any(series_or_index) -> pd.DatetimeIndex:
    """
    Robust date parsing:
      - Accept 'YYYY-MM-DD' strings
      - Accept 'YYYYMMDD' integers/strings
      - Strip timezones
      - Guarantee tz-naive datetime64[ns]
    """
    idx = pd.Index(series_or_index)

    # Already datetime-like?
    if isinstance(idx, pd.DatetimeIndex) or _is_dt(idx):
        di = pd.to_datetime(idx)
    else:
        # General parse
        di = pd.to_datetime(idx, errors="coerce")
        # If many NaT, try explicit YYYYMMDD
        if di.isna().mean() > 0.2:
            di = pd.to_datetime(idx.astype(str), format="%Y%m%d", errors="coerce")

    # Strip timezone if present
    try:
        di = di.tz_localize(None)
    except Exception:
        pass

    if di.isna().any():
        n_bad = int(di.isna().sum())
        print(f"[WARN] Dropping {n_bad} rows with unparseable dates")
        di = di[~di.isna()]

    return pd.DatetimeIndex(di)


def _align_dates(left: pd.DataFrame | pd.Series, right: pd.DataFrame) -> tuple[pd.Series | pd.DataFrame, pd.DataFrame]:
    li = _normalize_index_any(left.index)
    ri = _normalize_index_any(right.index)
    left = left.copy()
    right = right.copy()
    left.index = li
    right.index = ri
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise RuntimeError(
            "No overlapping dates between inputs after normalization. "
            "Check that both CSVs cover the same period and that 'Date' formats are parseable."
        )
    return left.loc[common], right.loc[common]


def _px_dates(idx) -> list:
    """Cast to plain Python datetimes for Kaleido static export stability."""
    di = pd.DatetimeIndex(idx)
    return di.to_pydatetime().tolist()


# ---------------------------
# IO
# ---------------------------
def load_portfolio() -> pd.Series:
    p = DATA / "portfolio_returns.csv"
    _req(p, "portfolio_returns.csv")
    df = pd.read_csv(p)
    if "Date" not in df or "Portfolio" not in df:
        raise ValueError("portfolio_returns.csv must have columns: Date, Portfolio")
    idx = _normalize_index_any(df["Date"])
    s = pd.Series(df["Portfolio"].astype(float).values, index=idx, name="Portfolio")
    s = s.sort_index()
    return s


def load_factors() -> pd.DataFrame:
    f = DATA / "factors.csv"
    _req(f, "factors.csv")
    df = pd.read_csv(f)
    if "Date" not in df:
        raise ValueError("factors.csv must have a Date column")
    idx = _normalize_index_any(df["Date"])
    df = df.drop(columns=["Date"], errors="ignore")
    # Ensure all needed columns exist and are numeric
    for c in ["RF", *FACTOR_COLS]:
        if c not in df.columns:
            raise ValueError(f"factors.csv missing column: {c}")
    df = df.astype(float)
    df.index = idx
    df = df.sort_index()
    return df


def load_attribution() -> pd.Series:
    a = DATA / "portfolio_attribution.csv"
    _req(a, "portfolio_attribution.csv")
    row = pd.read_csv(a).iloc[0].to_dict()
    out: Dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return pd.Series(out, dtype=float)


# ---------------------------
# Core utilities
# ---------------------------
def excess_returns(pf: pd.Series, fac: pd.DataFrame) -> pd.Series:
    pf2, fac2 = _align_dates(pf.to_frame(), fac[["RF"]])
    df = pd.concat([pf2, fac2], axis=1).dropna()
    if df.empty:
        raise RuntimeError("Excess returns join is empty after alignment.")
    ex = (df["Portfolio"] - df["RF"]).rename("Excess")
    return ex


def fitted_excess(excess: pd.Series, X: pd.DataFrame, betas: Dict[str, float], alpha: float) -> pd.Series:
    beta_vec = pd.Series({c: betas.get(f"beta_{c}", 0.0) for c in FACTOR_COLS}, index=FACTOR_COLS, dtype=float)
    # Align X to excess (and vice versa)
    X2, _ = _align_dates(X, excess.to_frame())
    ex2, _ = _align_dates(excess.to_frame(), X)
    fit = alpha + (X2[FACTOR_COLS] @ beta_vec)
    fit = fit.reindex(ex2.index)
    return fit


def residuals_series(excess: pd.Series, fit: pd.Series) -> pd.Series:
    r = (excess - fit).dropna()
    return r.reindex(excess.index)


def write(fig: go.Figure, png_path: Path, html_path: Path, width: int = 1100, height: int = 650) -> None:
    fig.update_layout(template=PLOTLY_TEMPLATE, colorway=COLORWAY)
    try:
        fig.write_image(str(png_path), format="png", width=width, height=height)
    except Exception as e:
        print(f"[WARN] PNG export skipped ({e}). HTML saved instead: {html_path.name}")
    fig.write_html(str(html_path), include_plotlyjs="cdn")


# ---------------------------
# Visuals
# ---------------------------
def fig_cumulative_returns(pf: pd.Series, fac: pd.DataFrame) -> go.Figure:
    pf2, fac2 = _align_dates(pf.to_frame(), fac[["MKT", "RF"]])
    df = pd.concat([pf2, fac2], axis=1).dropna()
    if df.empty:
        raise RuntimeError("Cumulative: empty join after date normalization.")
    df["MktTotal"] = df["RF"] + df["MKT"]
    df["Cum_Portfolio"] = (1.0 + df["Portfolio"]).cumprod()
    df["Cum_Market"] = (1.0 + df["MktTotal"]).cumprod()

    xs = _px_dates(df.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=df["Cum_Portfolio"], name="Portfolio ($1→)", mode="lines"))
    fig.add_trace(go.Scatter(x=xs, y=df["Cum_Market"], name="Market (RF+MKT)", mode="lines"))
    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(type="date", range=[xs[0], xs[-1]])
    return fig


def fig_rolling_betas(excess: pd.Series, fac: pd.DataFrame, window: int) -> go.Figure:
    X = fac[FACTOR_COLS]
    X2, _ = _align_dates(X, excess.to_frame())
    ex2 = excess.reindex(X2.index).dropna()
    if len(ex2) < window + 5:
        raise RuntimeError(f"Rolling betas: not enough overlapping rows ({len(ex2)}) for window={window}.")

    records = []
    idx = ex2.index
    for i in range(window, len(idx)):
        sl = idx[i - window:i]
        Xw = X2.loc[sl]
        Yw = ex2.loc[sl]
        Xc = np.column_stack([np.ones(len(Xw)), Xw.values])
        try:
            betahat, *_ = np.linalg.lstsq(Xc, Yw.values, rcond=None)  # [const, betas...]
            row = {"Date": sl[-1]}
            for j, c in enumerate(FACTOR_COLS):
                row[c] = betahat[j + 1]
            records.append(row)
        except Exception:
            continue

    roll = pd.DataFrame(records).set_index("Date").sort_index()
    if roll.empty:
        raise RuntimeError("Rolling beta table is empty; insufficient data for the chosen window.")

    xs = _px_dates(roll.index)
    fig = go.Figure()
    for c in FACTOR_COLS:
        if c in roll.columns:
            fig.add_trace(go.Scatter(x=xs, y=roll[c], name=c, mode="lines"))
    fig.update_layout(
        title=f"Rolling {window}-Day Factor Betas",
        xaxis_title="Date",
        yaxis_title="Beta",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(type="date", range=[xs[0], xs[-1]])
    return fig


def fig_residuals_hist(pf: pd.Series, fac: pd.DataFrame, att: pd.Series) -> go.Figure:
    ex = excess_returns(pf, fac)
    X = fac[FACTOR_COLS].reindex(ex.index)
    betas = {k: float(v) for k, v in att.items() if str(k).startswith("beta_")}
    alpha = float(att.get("alpha", 0.0))
    fit = fitted_excess(ex, X, betas, alpha)
    resid = residuals_series(ex, fit).dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=resid.values, name="Residuals", nbinsx=40, histnorm="probability density", opacity=0.85))
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        xs = np.linspace(resid.min(), resid.max(), 200)
        kde = gaussian_kde(resid.values)
        fig.add_trace(go.Scatter(x=xs, y=kde(xs), name="KDE", mode="lines"))
    except Exception:
        pass

    fig.add_vline(x=0.0, line_dash="dash", line_width=1)
    fig.update_layout(
        title="Residuals Distribution (Excess − Fitted)",
        xaxis_title="Residual",
        yaxis_title="Density",
        bargap=0.02,
        hovermode="x",
    )
    return fig


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate PCAT visuals (Plotly).")
    parser.add_argument("--data-dir", type=str, default=str(DATA), help="Path to data directory")
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS), help="Path to reports directory")
    parser.add_argument("--rolling-window", type=int, default=ROLL_WINDOW_DEFAULT, help="Rolling window (days) for betas")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    pf = load_portfolio()
    fac = load_factors()
    att = load_attribution()

    # Visibility on ranges
    print("[INFO] pf :", pf.index.min().date(),  "→", pf.index.max().date(),  "len:", len(pf))
    print("[INFO] fac:", fac.index.min().date(), "→", fac.index.max().date(), "len:", len(fac))

    ex = excess_returns(pf, fac)
    print("[INFO] ex :", ex.index.min().date(),  "→", ex.index.max().date(),  "len:", len(ex))

    # 1) Cumulative returns
    fig1 = fig_cumulative_returns(pf, fac)
    write(fig1, reports_dir / "cumulative_returns.png", reports_dir / "cumulative_returns.html")

    # 2) Rolling betas
    fig2 = fig_rolling_betas(ex, fac, window=args.rolling_window)
    write(fig2, reports_dir / f"rolling_betas_{args.rolling_window}d.png", reports_dir / f"rolling_betas_{args.rolling_window}d.html")

    # 3) Residuals histogram
    fig3 = fig_residuals_hist(pf, fac, att)
    write(fig3, reports_dir / "residuals_hist.png", reports_dir / "residuals_hist.html")

    print("Saved:")
    print(" -", reports_dir / "cumulative_returns.png")
    print(" -", reports_dir / f"rolling_betas_{args.rolling_window}d.png")
    print(" -", reports_dir / "residuals_hist.png")
    print(" (Interactive .html versions saved alongside each PNG)")


if __name__ == "__main__":
    main()
