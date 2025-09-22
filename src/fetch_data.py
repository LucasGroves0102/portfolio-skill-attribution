# src/fetch_data.py
import os, io, re, zipfile, requests
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL","MSFT","AMZN","NVDA","JPM","XOM"]
START, END = "2023-01-01", "2024-01-01"

FF5_ZIP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_ZIP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

def ensure_data_dir(): os.makedirs("data", exist_ok=True)

def fetch_prices(tickers, start, end) -> pd.DataFrame:
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
    if not isinstance(px.columns, pd.MultiIndex):
        prices.columns = tickers
    prices.index.name = "Date"
    return prices

def _read_zip_lines(url: str) -> list[str]:
    r = requests.get(url, timeout=30); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    return z.read(name).decode("latin1").splitlines()

def _block_from(lines: list[str], start_idx: int) -> str:
    """Join lines[start_idx: end_of_block] where end is first blank line after start_idx."""
    end = next((i for i in range(start_idx + 1, len(lines)) if lines[i].strip() == ""), len(lines))
    return "\n".join(lines[start_idx:end])

def _read_zip_data_block(url: str, header_regexes: list[str] | None = None) -> str:
    """
    Return the CSV data block starting at the header line up to the first blank line.
    Handles Ken French variants where the first column name (Date) is omitted.
    """
    lines = _read_zip_lines(url)

    header_regexes = header_regexes or [
        r"^Date\s*,",                 # normal: "Date, ..."
        r"^DATE\s*,",                 # uppercase
        r"^\s*,\s*Mkt[-\s_]*RF\b",    # variant: leading comma, no Date header
        r"^\s*,\s*Mom\b",             # momentum variant: leading comma, "Mom"
        r"^\s*,\s*MOM\b",
    ]

    hdr = None
    for pat in header_regexes:
        for i, ln in enumerate(lines):
            if re.search(pat, ln.strip(), flags=re.I):
                hdr = i
                break
        if hdr is not None:
            break

    if hdr is None:
        # As a last resort, if we see a data-looking line (8-digit yyyymmdd) just before many commas,
        # back up one line and treat it as headerless CSV with our own header injected later.
        try:
            first_data = next(i for i, ln in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", ln))
            hdr = first_data - 1 if first_data > 0 else first_data
        except StopIteration:
            preview = "\n".join(lines[:30])
            raise ValueError(
                f"Could not find CSV header in {url!r}. First lines preview:\n{preview}"
            )

    data_lines = []
    for ln in lines[hdr:]:
        if ln.strip() == "":
            break
        data_lines.append(ln)
    return "\n".join(data_lines)


def _parse_ff5(_: list[str] | None = None) -> pd.DataFrame:
    txt = _read_zip_data_block(
        FF5_ZIP,
        header_regexes=[
            r"^Date\s*,",
            r"^DATE\s*,",
            r"^\s*,\s*Mkt[-\s_]*RF\b",  # handles ",Mkt-RF,SMB,..."
        ],
    )
    df = pd.read_csv(io.StringIO(txt), engine="python")
    # If the 'Date' header was omitted, rename first column to 'Date'
    if "Date" not in df.columns:
        cols = list(df.columns)
        cols[0] = "Date"
        df.columns = cols

    # Normalize header variants like "Mkt_RF" → "Mkt-RF"
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.rename(columns={c: c.strip().replace("_", "-") for c in df.columns})

    required = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA","RF"]
    # If market column is named oddly, try to map it
    if "Mkt-RF" not in df.columns:
        candidates = [c for c in df.columns if re.fullmatch(r"(?i)mkt[-\s_]*rf", c)]
        if candidates:
            df = df.rename(columns={candidates[0]: "Mkt-RF"})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FF5 missing columns: {missing}. Got {list(df.columns)}")

    df = df[required].rename(columns={"Mkt-RF": "MKT"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df  # Date, MKT, SMB, HML, RMW, CMA


def _parse_mom(_: list[str] | None = None) -> pd.DataFrame:
    # Momentum files also sometimes omit 'Date' in the header and start with ",Mom"
    txt = _read_zip_data_block(
        MOM_ZIP,
        header_regexes=[
            r"^Date\s*,",
            r"^DATE\s*,",
            r"^\s*,\s*Mom\b",
            r"^\s*,\s*MOM\b",
        ],
    )
    mom = pd.read_csv(io.StringIO(txt), engine="python")
    mom = mom.rename(columns=lambda c: str(c).strip())

    # If Date header is missing, set first column as Date
    if "Date" not in mom.columns and "DATE" not in mom.columns:
        cols = list(mom.columns)
        cols[0] = "Date"
        mom.columns = cols
    mom = mom.rename(columns={"DATE": "Date"})

    # Find the momentum column (second column if unnamed properly)
    if "MOM" in mom.columns:
        mom_col = "MOM"
    elif "Mom" in mom.columns:
        mom_col = "Mom"
    else:
        candidates = [c for c in mom.columns if c != "Date"]
        if not candidates:
            raise ValueError(f"Could not find MOM column in MOM file. Got {list(mom.columns)}")
        mom_col = candidates[0]

    mom = mom[["Date", mom_col]].rename(columns={mom_col: "MOM"})
    mom["Date"] = pd.to_datetime(mom["Date"].astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
    mom["MOM"] = pd.to_numeric(mom["MOM"], errors="coerce")
    mom = mom.dropna(subset=["Date"]).reset_index(drop=True)
    return mom

def fetch_factors() -> pd.DataFrame:
    f5  = _parse_ff5(None)
    mom = _parse_mom(None)

    # Ensure numeric, convert percent → decimal
    for c in ["MKT","SMB","HML","RMW","CMA"]:
        f5[c] = pd.to_numeric(f5[c], errors="coerce") / 100.0
    mom["MOM"] = pd.to_numeric(mom["MOM"], errors="coerce") / 100.0

    # Inner merge on normalized dates
    factors = pd.merge(f5, mom, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    factors["RF"] = pd.to_numeric(factors["RF"], errors="coerce")/100.0
    # Sanity check to avoid silently writing an empty file
    if factors.empty:
        # Helpful diagnostics if something changes upstream
        f5_range  = (f5["Date"].min(),  f5["Date"].max())
        mom_range = (mom["Date"].min(), mom["Date"].max())
        raise RuntimeError(
            f"No overlapping dates between FF5 {f5_range} and MOM {mom_range}. "
            f"Sample FF5 head dates: {f5['Date'].head(3).tolist()} | "
            f"MOM head dates: {mom['Date'].head(3).tolist()}"
        )

    return factors[["Date","MKT","SMB","HML","RMW","CMA","MOM","RF"]]

def main():
    ensure_data_dir()
    print(f"Downloading prices for {len(TICKERS)} tickers...")
    fetch_prices(TICKERS, START, END).to_csv("data/prices.csv")
    print("Saved: data/prices.csv")
    print("Downloading Fama–French factors...")
    fetch_factors().to_csv("data/factors.csv", index=False)
    print("Saved: data/factors.csv")
    print("Done.")

if __name__ == "__main__":
    main()
