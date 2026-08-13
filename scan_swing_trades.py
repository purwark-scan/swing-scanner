"""
Daily Swing Trade Scanner — Nifty 500
Implements the rule set from "Complete Guide to Swing Trading" by Jaynesh Kasliwal:

  - 21 EMA > 50 EMA > 200 EMA        (trend stack)
  - 21 EMA sloping up over last 10d  (uptrend confirmation)
  - Stage 2: price > rising 150-day (~30wk) MA
  - HH-HL structure (last 2 swing highs/lows both rising)
  - Near 52-week high (within NEAR_52W_PCT of it)
  - Volume expansion (latest volume > VOL_MULT x 50-day avg volume)
  - Relative Strength > 0 vs NIFTY 50 (65-period, per the guide's settings)
  - MACD bullish (12,26,9): MACD line > signal, histogram > 0

Splits results into F&O and Non-F&O buckets and prints/saves the
top N candidates from each, ranked by a composite momentum score.

REQUIREMENTS
    pip install yfinance pandas numpy curl_cffi --break-system-packages

IMPORTANT — Yahoo Finance blocks cloud/datacenter IPs (GitHub Actions runners,
AWS, etc.) as of mid-2024. This script routes yfinance through curl_cffi with
a browser TLS fingerprint to work around that. If you still see every symbol
fail with empty data, Yahoo's blocklist has likely updated again — check for
a newer yfinance release or an alternate data source (e.g. nsepython).

USAGE
    python scan_swing_trades.py                 # scan full Nifty 500
    python scan_swing_trades.py --limit 50       # quick test on first 50
    python scan_swing_trades.py --top 5          # top N per bucket (default 5)

SCHEDULING (pick one — this script does NOT run itself):
  1. Linux/Mac cron (runs 4:00 PM IST on weekdays, after market close):
       30 10 * * 1-5 cd /path/to/swing_scanner && /usr/bin/python3 scan_swing_trades.py >> scan_log.txt 2>&1
       (10:30 UTC = 16:00 IST)

  2. Windows Task Scheduler:
       Create a daily trigger at 16:00 IST, action = python.exe scan_swing_trades.py

  3. GitHub Actions (free, no machine needed) — see scan.yml example
     provided alongside this script; runs on GitHub's servers on a cron
     schedule and can email/commit the output CSV automatically.

NOTES
  - Data source is Yahoo Finance via yfinance (delayed, free, no API key).
    Good enough for end-of-day swing scans; not for intraday/live trading.
  - The Nifty 500 & F&O lists are pulled live from NSE's public archives.
    If NSE's site structure changes or blocks the request, the script
    falls back to a bundled static list (may go stale over time --
    update FALLBACK_FNO_LIST below periodically).
  - This is a technical screener, not investment advice. Position sizing,
    stop-loss placement, and the final buy/no-buy decision are yours.
"""

import argparse
import io
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

def _make_yf_session():
    """Yahoo Finance blocks plain requests from cloud/datacenter IPs
    (GitHub Actions, AWS, etc.) since mid-2024. curl_cffi impersonates a
    real browser's TLS fingerprint, which routes around that block."""
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except ImportError:
        print("WARNING: curl_cffi not installed -- falling back to plain "
              "requests. If running on a cloud runner (GitHub Actions etc.), "
              "this will likely get blocked by Yahoo. Run: pip install curl_cffi")
        return None

YF_SESSION = _make_yf_session()

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
FNO_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

# Small static fallback in case the live NSE fetch fails (network block, etc).
# This is NOT the full F&O list -- just enough for the script to still run.
FALLBACK_FNO_LIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "MARUTI", "TITAN", "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "ADANIENT",
    "ULTRACEMCO", "NTPC", "POWERGRID", "HCLTECH", "WIPRO", "M&M",
    "HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT", "DRREDDY", "CIPLA", "GRASIM",
    "JSWSTEEL", "COALINDIA", "ONGC", "BPCL", "IOC", "TECHM", "ASIANPAINT",
    "NESTLEIND", "DIVISLAB", "BRITANNIA", "SHREECEM", "HINDALCO", "APOLLOHOSP",
    "SBILIFE", "HDFCLIFE", "INDUSINDBK", "BAJAJFINSV", "UPL", "TATACONSUM",
]

NEAR_52W_PCT = 0.10      # within 10% of 52-week high
VOL_MULT = 1.5           # breakout volume vs 50-day avg volume
RS_PERIOD = 65           # matches "bharattrader" RS setting mentioned in the guide
HH_HL_LOOKBACK = 60      # bars used to detect swing structure


def fetch_csv(url, col_candidates):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    for c in col_candidates:
        if c in df.columns:
            return df[c].astype(str).str.strip().tolist()
    raise ValueError(f"None of {col_candidates} found in columns {df.columns}")


def get_nifty500_symbols():
    try:
        syms = fetch_csv(NIFTY500_URL, ["Symbol"])
        print(f"Fetched {len(syms)} Nifty 500 symbols from NSE.")
        return syms
    except Exception as e:
        print(f"WARNING: could not fetch Nifty 500 list live ({e}). "
              f"Falling back to the static F&O-only list for this run.")
        return FALLBACK_FNO_LIST


def get_fno_symbols():
    try:
        syms = fetch_csv(FNO_URL, ["SYMBOL", "Symbol"])
        syms = [s for s in syms if s.isupper() and s.isalnum() is False or s.isalpha()]
        print(f"Fetched {len(syms)} F&O symbols from NSE.")
        return set(syms)
    except Exception as e:
        print(f"WARNING: could not fetch F&O list live ({e}). Using static fallback.")
        return set(FALLBACK_FNO_LIST)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def detect_hh_hl(close, lookback=HH_HL_LOOKBACK):
    """Rough swing-structure check: split the lookback window into two
    halves, compare the max (swing high) and min (swing low) of each half.
    True HH-HL requires both to rise from the first half to the second."""
    window = close[-lookback:]
    if len(window) < lookback:
        return False
    half = lookback // 2
    first_half, second_half = window[:half], window[half:]
    hh = second_half.max() > first_half.max()
    hl = second_half.min() > first_half.min()
    return bool(hh and hl)


def compute_macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def score_stock(df, nifty_close):
    close = df["Close"]
    volume = df["Volume"]
    if len(close) < 210:
        return None

    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    ma150 = close.rolling(150).mean()

    last = close.iloc[-1]
    e21, e50, e200 = ema21.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]
    ma150_now, ma150_prev = ma150.iloc[-1], ma150.iloc[-10]

    # --- Rule 1: EMA stack ---
    ema_stack = e21 > e50 > e200

    # --- Rule 2: 21 EMA sloping up ---
    ema21_slope_up = ema21.iloc[-1] > ema21.iloc[-10]

    # --- Rule 3: Stage 2 (price above rising 150-day MA) ---
    stage2 = (last > ma150_now) and (ma150_now > ma150_prev)

    # --- Rule 4: HH-HL structure ---
    hh_hl = detect_hh_hl(close)

    # --- Rule 5: near 52-week high ---
    high_52w = close[-252:].max() if len(close) >= 252 else close.max()
    near_52w_high = last >= high_52w * (1 - NEAR_52W_PCT)

    # --- Rule 6: volume expansion ---
    avg_vol_50 = volume.rolling(50).mean().iloc[-1]
    vol_expansion = volume.iloc[-1] > VOL_MULT * avg_vol_50 if avg_vol_50 > 0 else False
    # also check breakout happened on above-average volume in last 5 bars
    recent_vol_ok = (volume[-5:] > avg_vol_50).any()

    # --- Rule 7: Relative Strength vs Nifty ---
    aligned = pd.concat([close, nifty_close], axis=1, join="inner").dropna()
    aligned.columns = ["stock", "nifty"]
    if len(aligned) < RS_PERIOD + 1:
        rs_positive = False
        rs_value = np.nan
    else:
        stock_ret = aligned["stock"].iloc[-1] / aligned["stock"].iloc[-RS_PERIOD] - 1
        nifty_ret = aligned["nifty"].iloc[-1] / aligned["nifty"].iloc[-RS_PERIOD] - 1
        rs_value = stock_ret - nifty_ret
        rs_positive = rs_value > 0

    # --- Rule 8: MACD bullish ---
    macd_line, signal_line, hist = compute_macd(close)
    macd_bullish = (macd_line.iloc[-1] > signal_line.iloc[-1]) and (hist.iloc[-1] > 0)

    rules = {
        "ema_stack": ema_stack,
        "ema21_slope_up": ema21_slope_up,
        "stage2": stage2,
        "hh_hl": hh_hl,
        "near_52w_high": near_52w_high,
        "vol_expansion_or_recent": vol_expansion or recent_vol_ok,
        "rs_positive": rs_positive,
        "macd_bullish": macd_bullish,
    }
    passed = sum(rules.values())
    total = len(rules)

    # Composite momentum score for ranking (used only to sort candidates
    # that already pass the mandatory filter below)
    pct_from_high = (last / high_52w - 1) * 100
    score = (passed / total) * 100 + (rs_value * 100 if not np.isnan(rs_value) else 0)

    return {
        "close": round(float(last), 2),
        "pct_from_52w_high": round(float(pct_from_high), 2),
        "rs_vs_nifty_pct": round(float(rs_value) * 100, 2) if not np.isnan(rs_value) else None,
        "rules_passed": f"{passed}/{total}",
        "rules_detail": rules,
        "score": round(float(score), 2),
    }


def run_scan(symbols, fno_set, top_n, limit=None):
    import yfinance as yf

    if limit:
        symbols = symbols[:limit]

    print(f"Downloading NIFTY 50 index data for RS calc...")
    nifty = yf.download("^NSEI", period="2y", progress=False, auto_adjust=True,
                         session=YF_SESSION)
    if nifty.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data even for ^NSEI. This almost "
            "always means Yahoo is blocking this server's IP. Confirm "
            "curl_cffi is installed and try again; if it still fails, "
            "Yahoo's blocklist may have changed and a different data "
            "source (e.g. nsepython) is needed."
        )
    nifty_close = nifty["Close"]
    if isinstance(nifty_close, pd.DataFrame):
        nifty_close = nifty_close.iloc[:, 0]

    results = []
    failed = []
    for i, sym in enumerate(symbols):
        ticker = f"{sym}.NS"
        try:
            data = yf.download(ticker, period="2y", progress=False, auto_adjust=True,
                                session=YF_SESSION)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty or len(data) < 210:
                failed.append(sym)
                continue
            res = score_stock(data, nifty_close)
            if res is None:
                failed.append(sym)
                continue
            res["symbol"] = sym
            res["is_fno"] = sym in fno_set
            results.append(res)
        except Exception as e:
            failed.append(sym)
        if (i + 1) % 25 == 0:
            print(f"  ...scanned {i+1}/{len(symbols)}")
        time.sleep(0.05)  # be polite to Yahoo's endpoint

    print(f"Scan complete. {len(results)} scored, {len(failed)} failed/skipped.")

    df = pd.DataFrame(results)
    if df.empty:
        print("No results.")
        return df

    # Mandatory filter: require at least 6 of 8 rules to pass, and the
    # non-negotiable ones (EMA stack, near 52w high, RS positive) must hold.
    def qualifies(row):
        d = row["rules_detail"]
        must_have = d["ema_stack"] and d["near_52w_high"] and d["rs_positive"]
        return must_have and int(row["rules_passed"].split("/")[0]) >= 6

    df["qualifies"] = df.apply(qualifies, axis=1)
    qualified = df[df["qualifies"]].sort_values("score", ascending=False)

    fno_picks = qualified[qualified["is_fno"]].head(top_n)
    non_fno_picks = qualified[~qualified["is_fno"]].head(top_n)

    # Near-misses: best-scoring stocks that didn't fully qualify, so you
    # always have visibility into "closest to a setup" even on quiet days.
    near_miss = df[~df["qualifies"]].sort_values("score", ascending=False)
    near_miss_fno = near_miss[near_miss["is_fno"]].head(top_n)
    near_miss_non_fno = near_miss[~near_miss["is_fno"]].head(top_n)

    # Rule-level pass-rate across the whole scanned universe, so you can see
    # WHICH rule is the bottleneck on a zero-result day (e.g. "near_52w_high"
    # passing on only 4% of stocks tells you the market is in a pullback).
    rule_names = list(df["rules_detail"].iloc[0].keys())
    pass_rates = {r: round(df["rules_detail"].apply(lambda d: d[r]).mean() * 100, 1)
                  for r in rule_names}

    return df, fno_picks, non_fno_picks, near_miss_fno, near_miss_non_fno, pass_rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit universe for quick testing")
    parser.add_argument("--top", type=int, default=5, help="number of picks per bucket")
    args = parser.parse_args()

    print(f"=== Swing Trade Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    symbols = get_nifty500_symbols()
    fno_set = get_fno_symbols()

    full_df, fno_picks, non_fno_picks, near_miss_fno, near_miss_non_fno, pass_rates = \
        run_scan(symbols, fno_set, args.top, args.limit)

    out_file = f"scan_results_{datetime.now().strftime('%Y%m%d')}.csv"
    full_df.drop(columns=["rules_detail"]).to_csv(out_file, index=False)
    print(f"\nFull results saved to {out_file}\n")

    print("--- RULE PASS-RATE ACROSS SCANNED UNIVERSE (diagnostic) ---")
    for rule, pct in pass_rates.items():
        print(f"  {rule:30s} {pct:5.1f}% of stocks pass")
    print()

    cols = ["symbol", "close", "pct_from_52w_high", "rs_vs_nifty_pct", "rules_passed", "score"]

    print(f"--- TOP {args.top} F&O SWING CANDIDATES (fully qualified) ---")
    if fno_picks.empty:
        print("  None qualified today.")
    else:
        print(fno_picks[cols].to_string(index=False))

    print(f"\n--- TOP {args.top} NON-F&O SWING CANDIDATES (fully qualified) ---")
    if non_fno_picks.empty:
        print("  None qualified today.")
    else:
        print(non_fno_picks[cols].to_string(index=False))

    print(f"\n--- CLOSEST F&O NEAR-MISSES (didn't fully qualify, but ranked) ---")
    if near_miss_fno.empty:
        print("  (no scored F&O stocks)")
    else:
        print(near_miss_fno[cols].to_string(index=False))

    print(f"\n--- CLOSEST NON-F&O NEAR-MISSES (didn't fully qualify, but ranked) ---")
    if near_miss_non_fno.empty:
        print("  (no scored non-F&O stocks)")
    else:
        print(near_miss_non_fno[cols].to_string(index=False))


if __name__ == "__main__":
    main()
