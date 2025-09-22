# src/quick_positons.py
import pandas as pd

def main():
    px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    tickers = list(px.columns)
    w = 1.0 / len(tickers)
    pos = pd.DataFrame(index=px.index, columns=tickers)
    pos[:] = w  # equal-weight each day
    pos.to_csv("data/positions.csv")
    print("Saved data/positions.csv with columns:", tickers)

if __name__ == "__main__":
    main()