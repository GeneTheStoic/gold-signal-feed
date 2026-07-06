"""
feed_publisher_v2.py

Daily feed publisher for the gold options signal utility (Product #2).
GitHub Actions edition: this script ONLY runs the pipeline and writes the
feed file. Committing and pushing is handled by the workflow (.github/
workflows/feed.yml), so there is no git logic in here.

WHAT IT DOES
  1. Runs the full signal pipeline (identical math to gold_signals_macro.py).
  2. Takes the LATEST settled row.
  3. Applies the arrow logic SERVER-SIDE:
       BUY  when zscore_ENSEMBLE >= +0.5 AND regime_favorable AND macro_favorable
       NONE otherwise
  4. Writes ONE tiny feed file: feed/gold_signal.csv (fixed 2-line format).
  5. On any failure, optionally sends a Telegram alert, then exits non-zero
     so the workflow run is marked failed (and GitHub emails you).

FEED FORMAT (exactly 2 lines, comma separated)
  schema,date,signal,ensemble,regime_favorable,macro_favorable,trend_er,generated_utc
  1,2026.07.03,BUY,0.8231,1,1,0.4123,2026-07-06T06:00:12Z

OPTIONAL SECRETS (set in the repo: Settings > Secrets and variables > Actions)
  TG_BOT_TOKEN     Telegram bot token for failure alerts
  TG_CHAT_ID       Telegram chat id for failure alerts

EXIT CODES
  0 = feed written
  1 = pipeline or publish failure
"""

import os
import sys
import threading
import warnings
from datetime import datetime, timezone

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance pandas numpy")
    sys.exit(1)

# ------------------------- CONFIG -------------------------
START_DATE        = "2013-01-01"
ZSCORE_WINDOW     = 252
EGARCH_WINDOW     = 252
DOWNLOAD_TIMEOUT  = 30
SIGNAL_DIRECTIONS = {"VRP": +1, "SKEW": -1, "PCR": -1, "TERM": +1}

ENSEMBLE_THRESHOLD = 0.5
SCHEMA_VERSION     = 1

REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
FEED_DIR  = os.path.join(REPO_DIR, "feed")
FEED_FILE = os.path.join(FEED_DIR, "gold_signal.csv")

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "")


# ------------------------- ALERTING -------------------------
def telegram_alert(msg):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TG_CHAT_ID, "text": msg}).encode()
        urllib.request.urlopen(url, data=data, timeout=15)
    except Exception:
        pass


def fail(msg):
    print(f"FAIL: {msg}")
    telegram_alert(f"GOLD FEED FAILED: {msg}")
    sys.exit(1)


# ------------------------- DATA DOWNLOAD -------------------------
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


def _fetch_history(ticker):
    box = {}

    def _worker():
        try:
            t = yf.Ticker(ticker)
            box["df"] = t.history(start=START_DATE, end=END_DATE,
                                  auto_adjust=True, timeout=DOWNLOAD_TIMEOUT,
                                  raise_errors=True)
        except Exception as e:
            box["err"] = e

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(DOWNLOAD_TIMEOUT + 5)
    if th.is_alive():
        raise TimeoutError(f"download exceeded {DOWNLOAD_TIMEOUT + 5}s hard timeout")
    if "err" in box:
        raise box["err"]
    return box.get("df")


def dl(ticker, name, retries=2):
    for attempt in range(retries + 1):
        try:
            df = _fetch_history(ticker)
            if df is None or len(df) < 50:
                print(f"  XX {name} ({ticker}): insufficient/empty data")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df["Close"].rename(name)
            s.index = s.index.tz_localize(None)
            print(f"  OK {name} ({ticker}): {len(s)} rows")
            return s
        except Exception as e:
            if attempt < retries:
                print(f"  .. {name} ({ticker}) retry {attempt + 1}: {e}")
                continue
            print(f"  XX {name} ({ticker}): {e}")
            return None


# ------------------------- PIPELINE (identical math) -------------------------
def egarch_variance_series(prices, window=252):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    result = pd.Series(np.nan, index=prices.index)
    alpha = 2.0 / (20 + 1)
    gamma = -0.10
    price_idx = list(prices.index)
    ret_idx = list(log_ret.index)
    for i in range(window, len(log_ret)):
        r = log_ret.iloc[i - window: i].values
        v = np.var(r) + 1e-12
        for rt in r:
            z = rt / np.sqrt(v)
            leverage = abs(z) + gamma * z
            v = v * np.exp(alpha * (leverage - np.sqrt(2.0 / np.pi)))
            v = max(v, 1e-12)
        orig_idx = ret_idx[i]
        try:
            pos = price_idx.index(orig_idx)
            result.iloc[pos] = v * 252.0
        except ValueError:
            pass
    return result


def zscore(s, w, min_p=63):
    m = s.rolling(w, min_periods=min_p).mean()
    sd = s.rolling(w, min_periods=min_p).std()
    return ((s - m) / sd.where(sd > 1e-10, np.nan)).clip(-4, 4)


def efficiency_ratio(s, n=20):
    change = s.diff(n).abs()
    vol = s.diff().abs().rolling(n).sum()
    return change / vol.where(vol > 0, np.nan)


def run_pipeline():
    print("[1] Downloading gold + options data...")
    GLD = dl("GLD", "GLD")
    GVZ = dl("^GVZ", "GVZ")
    VIX = dl("^VIX", "VIX")
    VIX3M = dl("^VIX3M", "VIX3M")
    SLV = dl("SLV", "SLV")

    print("[1b] Downloading macro regime data...")
    DXY = dl("DX-Y.NYB", "DXY")
    if DXY is None:
        DXY = dl("UUP", "DXY(UUP proxy)")
    TIP = dl("TIP", "TIP")
    TNX = dl("^TNX", "TNX")

    if GLD is None:
        fail("Cannot download GLD (core series).")
    if GVZ is None:
        fail("Cannot download GVZ (signals depend on it).")

    df = pd.DataFrame({"GLD": GLD})
    for name, s in [("GVZ", GVZ), ("VIX", VIX), ("VIX3M", VIX3M), ("SLV", SLV),
                    ("DXY", DXY), ("TIP", TIP), ("TNX", TNX)]:
        if s is not None:
            df = df.join(s.rename(name), how="left")

    df = df.sort_index().loc[START_DATE:]
    df.index = pd.to_datetime(df.index)
    for col in ["GVZ", "VIX", "VIX3M", "DXY", "TIP", "TNX"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    print("[2] Signals...")
    print("  EGARCH(1,1)... (~30 seconds)")
    df["EGARCH_VAR"] = egarch_variance_series(df["GLD"], window=EGARCH_WINDOW)
    df["signal_VRP"] = (df["GVZ"] / 100.0) ** 2 - df["EGARCH_VAR"]

    df["signal_SKEW"] = df["GVZ"]
    if "SLV" in df.columns:
        gs = df["GLD"] / df["SLV"]
        gs_ma = gs.rolling(63, min_periods=20).mean()
        df["signal_SKEW"] = df["GVZ"] + ((gs - gs_ma) / gs_ma) * 10

    if "VIX" in df.columns:
        df["signal_PCR"] = df["GVZ"] - df["VIX"]
    else:
        df["signal_PCR"] = df["GVZ"]

    gld_ret = np.log(df["GLD"] / df["GLD"].shift(1))
    rv21 = gld_ret.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
    df["signal_TERM"] = df["GVZ"] - rv21

    print("[3] Z-scores + ensemble...")
    parts = []
    for sig in ["VRP", "SKEW", "PCR", "TERM"]:
        df[f"zscore_{sig}"] = zscore(df[f"signal_{sig}"], ZSCORE_WINDOW)
        parts.append(df[f"zscore_{sig}"] * SIGNAL_DIRECTIONS[sig])
    df["zscore_ENSEMBLE"] = pd.concat(parts, axis=1).mean(axis=1)

    print("[4] Regimes...")
    gvz_ma = df["GVZ"].rolling(252, min_periods=63).mean()
    high_gvz = df["GVZ"] > gvz_ma * 1.1
    if "VIX" in df.columns:
        vix_ma = df["VIX"].rolling(252, min_periods=63).mean()
        df["regime_favorable"] = (high_gvz | (df["VIX"] > vix_ma * 0.9)).astype(int)
    else:
        df["regime_favorable"] = high_gvz.astype(int)

    df["trend_ER"] = efficiency_ratio(df["GLD"], 20).clip(0, 1)
    g50 = df["GLD"].rolling(50).mean()
    g200 = df["GLD"].rolling(200, min_periods=100).mean()
    df["gold_uptrend"] = ((df["GLD"] > g50) & (g50 >= g200)).astype(int)

    if "DXY" in df.columns:
        df["dxy_weak"] = (df["DXY"] < df["DXY"].rolling(50).mean()).astype(int)
    else:
        print("  WARN: no DXY -> dxy_weak permissive (1)")
        df["dxy_weak"] = 1

    if "TIP" in df.columns:
        df["real_yield_down"] = (df["TIP"] > df["TIP"].rolling(50).mean()).astype(int)
    else:
        print("  WARN: no TIP -> real_yield_down permissive (1)")
        df["real_yield_down"] = 1

    df["macro_favorable"] = (
        (df["gold_uptrend"] == 1) &
        ((df["dxy_weak"] == 1) | (df["real_yield_down"] == 1))
    ).astype(int)

    return df


# ------------------------- FEED BUILD -------------------------
def build_feed_row(df):
    valid = df.dropna(subset=["zscore_ENSEMBLE"])
    if len(valid) == 0:
        fail("Pipeline produced no valid ensemble rows.")

    row = valid.iloc[-1]
    row_date = valid.index[-1]

    age_days = (pd.Timestamp.today().normalize() - row_date.normalize()).days
    if age_days > 5:
        fail(f"Latest settled row is {age_days} days old ({row_date.date()}). Not publishing.")

    ensemble = float(row["zscore_ENSEMBLE"])
    regime = int(row["regime_favorable"])
    macro = int(row["macro_favorable"])
    trend_er = float(row["trend_ER"]) if pd.notna(row["trend_ER"]) else 0.0

    signal = "BUY" if (ensemble >= ENSEMBLE_THRESHOLD and regime == 1 and macro == 1) else "NONE"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = "schema,date,signal,ensemble,regime_favorable,macro_favorable,trend_er,generated_utc"
    line = (f"{SCHEMA_VERSION},{row_date.strftime('%Y.%m.%d')},{signal},"
            f"{ensemble:.4f},{regime},{macro},{trend_er:.4f},{generated}")
    return header + "\n" + line + "\n", signal, row_date, ensemble


def write_feed(content):
    os.makedirs(FEED_DIR, exist_ok=True)
    with open(FEED_FILE, "w", newline="\n") as f:
        f.write(content)
    print(f"[5] Feed written: {FEED_FILE}")


# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Gold Signal Feed Publisher v2 (GitHub Actions)")
    print("=" * 60)
    try:
        data = run_pipeline()
        content, signal, row_date, ensemble = build_feed_row(data)
        print("\n--- FEED CONTENT ---")
        print(content.strip())
        print("--------------------")
        write_feed(content)
        print(f"\nDONE. {row_date.date()} -> {signal} (ensemble {ensemble:+.2f})")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        fail(f"Unhandled: {e}")
