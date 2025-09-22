# main.py
from __future__ import annotations
import argparse, importlib, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
for p in (SRC, DATA, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

# Ensure "src" is importable
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run(title: str, fn, *argv_like) -> None:
    t0 = time.time()
    print(f"\n[STEP] {title} …")
    if argv_like:
        bak = sys.argv[:]
        try:
            sys.argv = [fn.__module__] + list(argv_like)
            fn()
        finally:
            sys.argv = bak
    else:
        fn()
    print(f"[OK] {title} ({time.time()-t0:0.2f}s)")


def _maybe(modname: str, attr: str = "main"):
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, attr, None)
    except Exception as e:
        print(f"[SKIP] {modname}: {e}")
        return None


def pipeline(
    fetch: bool,
    make_positions: bool,
    build_portfolio: bool,
    attribute: bool,
    skill: bool,
    forecast: bool,
    visuals: bool,
    rolling_window: int | None,
    fc_model: str | None,
    fc_horizon: int | None,
    ewma_lambda: float | None,
):
    # 1) Fetch (prices + Fama–French + MOM)
    if fetch:
        fn = _maybe("fetch_data", "main")
        if fn: _run("Fetch prices & factors", fn)

    # 2) (Optional) quick, equal-weight positions
    if make_positions:
        fn = _maybe("quick_positons", "main") or _maybe("quick_positions", "main")
        if fn: _run("Create equal-weight positions", fn)

    # 3) Build portfolio_returns.csv
    if build_portfolio:
        fn = _maybe("portfolio_returns", "main")
        if fn: _run("Compute daily portfolio returns", fn)

    # 4) Attribution (6-factor + timing tests)
    if attribute:
        fn = _maybe("attribution", "main")
        if fn: _run("Run factor attribution & timing tests", fn)

    # 5) Skill metrics (Sharpe, TE, alpha IR, MDD, etc.)
    if skill:
        fn = _maybe("skill_metric", "main")
        if fn: _run("Compute skill metrics", fn)

    # 6) Forecast (factor-based)
    if forecast:
        fn = _maybe("forecast", "main")
        if fn:
            argv = []
            if fc_model:   argv += ["--model", fc_model]
            if fc_horizon: argv += ["--horizon", str(fc_horizon)]
            if ewma_lambda is not None: argv += ["--ewma-lambda", str(ewma_lambda)]
            _run(f"Forecast ({fc_model or 'ewma'})", fn, *argv)

    # 7) Visuals (PNGs + HTMLs)
    if visuals:
        fn = _maybe("visuals", "main")
        if fn:
            argv = ["--reports-dir", str(REPORTS)]
            if rolling_window: argv += ["--rolling-window", str(rolling_window)]
            _run(f"Generate visuals (rolling={rolling_window or 'default'})", fn, *argv)

    print("\n✅ Pipeline complete.")
    print(f"   Data   → {DATA}")
    print(f"   Reports→ {REPORTS}")


def main():
    ap = argparse.ArgumentParser(description="PCAT — Portfolio Skill Attribution end-to-end runner")
    ap.add_argument("--no-fetch",        action="store_true", help="Skip fetching prices/factors")
    ap.add_argument("--no-positions",    action="store_true", help="Skip generating equal-weight positions")
    ap.add_argument("--no-portfolio",    action="store_true", help="Skip portfolio_returns computation")
    ap.add_argument("--no-attribute",    action="store_true", help="Skip factor attribution & timing tests")
    ap.add_argument("--no-skill",        action="store_true", help="Skip skill_metrics")
    ap.add_argument("--no-forecast",     action="store_true", help="Skip forecast")
    ap.add_argument("--no-visuals",      action="store_true", help="Skip visuals")
    ap.add_argument("--rolling-window",  type=int, default=None, help="Rolling beta window (days)")

    ap.add_argument("--fc-model",        type=str, choices=["mean","ewma","var1"], default="ewma", help="Forecast model")
    ap.add_argument("--fc-horizon",      type=int, default=1, help="Forecast horizon (trading days)")
    ap.add_argument("--ewma-lambda",     type=float, default=0.94, help="EWMA decay (if fc-model=ewma)")

    args = ap.parse_args()

    pipeline(
        fetch=not args.no_fetch,
        make_positions=not args.no_positions,
        build_portfolio=not args.no_portfolio,
        attribute=not args.no_attribute,
        skill=not args.no_skill,
        forecast=not args.no_forecast,
        visuals=not args.no_visuals,
        rolling_window=args.rolling_window,
        fc_model=args.fc_model,
        fc_horizon=args.fc_horizon,
        ewma_lambda=args.ewma_lambda,
    )


if __name__ == "__main__":
    main()
