# src/data_io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

DATA.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)

# ---------- Generic helpers ----------
def _req(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {name} at: {path}")

# ---------- Prices ----------
def load_prices() -> pd.DataFrame:
    p = DATA / "prices.csv"
    _req(p, "prices.csv")
    return pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()

def save_prices(df: pd.DataFrame) -> None:
    (DATA / "prices.csv").write_text(df.to_csv())

# ---------- Factors ----------
def load_factors() -> pd.DataFrame:
    p = DATA / "factors.csv"
    _req(p, "factors.csv")
    return pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()

def save_factors(df: pd.DataFrame) -> None:
    (DATA / "factors.csv").write_text(df.to_csv(index=False))

# ---------- Positions ----------
def load_positions() -> pd.DataFrame:
    p = DATA / "positions.csv"
    _req(p, "positions.csv")
    return pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()

def save_positions(df: pd.DataFrame) -> None:
    (DATA / "positions.csv").write_text(df.to_csv())

# ---------- Portfolio returns ----------
def load_portfolio_returns() -> pd.Series:
    p = DATA / "portfolio_returns.csv"
    _req(p, "portfolio_returns.csv")
    return pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()["Portfolio"]

def save_portfolio_returns(s: pd.Series) -> None:
    s.to_frame(name="Portfolio").to_csv(DATA / "portfolio_returns.csv")
