import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def main():
    prices = pd.read_csv(DATA / "prices.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    positions = pd.read_csv(DATA / "positions.csv", parse_dates=["Date"]).set_index("Date").sort_index()

    rets = prices.pct_change()
    w = positions.reindex(rets.index).shift(1)  # use yesterday's weights (no look-ahead)

    # drop first row created by shift and compute portfolio return
    pf = (rets.iloc[1:] * w.iloc[1:]).sum(axis=1).to_frame(name="Portfolio")
    pf.to_csv(DATA / "portfolio_returns.csv")
    print("Saved", DATA / "portfolio_returns.csv", "| rows:", len(pf))

if __name__ == "__main__":
    main()
