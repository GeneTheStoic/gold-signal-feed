"""
feed_publisher_v3.py

Daily feed publisher for the gold options signal utility (Product #2).
GitHub Actions edition.

v3 CHANGE: the BUY/NONE decision now replicates the LONG-ENTRY STACK of
OptionsAlpha_ULTRA_FUNDED v4.5 (the validated, FTMO-passing configuration)
instead of the simple provisional rule (ensemble >= 0.5 AND regime AND macro)
used in v2. Gate for gate, mirrored from the EA source:

  1. Score gate        readiness score >= 3.0 (mirrors ComputeReadinessScore:
                       10-day avg ensemble, latest VRP, persistence, regime)
  2. Macro gate        macro_favorable == 1
  3. VRP gate          OFF (matches the live funded configuration)
  4. Trend filter      dual EMA 50/200 on gold, 5-state classification,
                       STRONG_UP boosts ensemble 1.3x, NEUTRAL/WEAK_DOWN
                       tighten the threshold (x1.5 / x2.5), downtrends block
  5. Regime + v19      regime_favorable required, but overridden to OK when
                       trend is STRONG_UP or WEAK_UP and ensemble >= 0.35
  6. RSI gate          RSI(14) daily, minimum depends on trend state
                       (40 strong-up, 45 weak-up, 40 neutral, 45 weak-down)
  7. Flash-crash       previous day range > 2.0 x ATR(14) blocks entry
  8. Agreement         at least 1 individual signal with directed z >= 0.5
  9. Threshold         filtered ensemble >= 0.20 (base, before trend adj.)

KNOWN PROXY CAVEAT: trend, RSI, ATR and flash-crash are computed on GLD
daily OHLC (the server has no broker XAUUSD feed). GLD tracks spot gold
closely; borderline days may rarely diverge from a broker-chart computation.

FEED FORMAT: unchanged from v2 (schema 1). The chart utility needs no update.
  schema,date,signal,ensemble,regime_favorable,macro_favorable,trend_er,generated_utc

EXIT CODES:  0 = feed written,  1 = failure (workflow marks the run red)
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

# ------------------------- PIPELINE CONFIG -------------------------
START_DATE        = "2013-01-01"
ZSCORE_WINDOW     = 252
EGARCH_WINDOW     = 252
DOWNLOAD_TIMEOUT  = 30
SIGNAL_DIRECTIONS = {"VRP": +1, "SKEW": -1, "PCR": -1, "TERM": +1}
SCHEMA_VERSION    = 1

# ------------------------- ENTRY STACK CONFIG ----------------------
# Mirrored from OptionsAlpha_ULTRA_FUNDED v4.5 inputs (long side only).
ENSEMBLE_LONG_THRESHOLD = 0.20
TRANSITION_MULT_NEUTRAL = 1.5
TRANSITION_MULT_WEAKDN  = 2.5
TREND_STRENGTH_MIN_PCT  = 1.5
FAST_MA_PERIOD          = 50
SLOW_MA_PERIOD          = 200
RSI_PERIOD              = 14
RSI_LONGMIN_STRONG_UP   = 40.0
RSI_LONGMIN_WEAK_UP     = 45.0
RSI_LONGMIN_NEUTRAL     = 40.0
RSI_LONGMIN_WEAK_DOWN   = 45.0
RSI_LONGMIN_DEFAULT     = 55.0
FLASHCRASH_ATR_MULT     = 2.0
ATR_PERIOD              = 14
INDIV_SIGNAL_THRESHOLD  = 0.50
MIN_SIGNALS_REQUIRED    = 1
REGIME_OVERRIDE_ENS     = 0.35
REGIME_OVERRIDE_WEAKUP  = True
USE_VRP_GATE            = False    # live funded config: OFF (more profit)
VRP_MIN_Z               = 0.0
SCORE_LOOKBACK          = 10
MIN_DEPLOY_SCORE        = 3.0
SCORE_ENS_HIGH          = 0.30
SCORE_ENS_MID           = 0.15
SCORE_VRP_HIGH          = 0.50
SCORE_PERS_HIGH         = 0.60
SCORE_PERS_MID          = 0.40

REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
FEED_DIR  = os.path.join(REPO_DIR, "feed")
FEED_FILE = os.path.join(FEED_DIR, "gold_signal.csv")

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "")

TREND_STRONG_UP, TREND_WEAK_UP, TREND_NEUTRAL = 2, 1, 0
TREND_WEAK_DOWN, TREND_STRONG_DOWN = -1, -2
TREND_NAMES = {2: "STRONG_UP", 1: "WEAK_UP", 0: "NEUTRAL",
               -1: "WEAK_DOWN", -2: "STRONG_DOWN"}


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


def dl_full(ticker, name, retries=2):
    """Full OHLC dataframe."""
    for attempt in range(retries + 1):
        try:
            df = _fetch_history(ticker)
            if df is None or len(df) < 50:
                print(f"  XX {name} ({ticker}): insufficient/empty data")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)
            print(f"  OK {name} ({ticker}): {len(df)} rows")
            return df
        except Exception as e:
            if attempt < retries:
                print(f"  .. {name} ({ticker}) retry {attempt + 1}: {e}")
                continue
            print(f"  XX {name} ({ticker}): {e}")
            return None


def dl(ticker, name, retries=2):
    """Close series only."""
    full = dl_full(ticker, name, retries)
    if full is None:
        return None
    return full["Close"].rename(name)


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


def wilder_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 1e-12, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def wilder_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def run_pipeline():
    print("[1] Downloading gold + options data...")
    gld_full = dl_full("GLD", "GLD")
    GVZ = dl("^GVZ", "GVZ")
    VIX = dl("^VIX", "VIX")
    VIX3M = dl("^VIX3M", "VIX3M")
    SLV = dl("SLV", "SLV")

    print("[1b] Downloading macro regime data...")
    DXY = dl("DX-Y.NYB", "DXY")
    if DXY is None:
        DXY = dl("UUP", "DXY(UUP proxy)")
    TIP = dl("TIP", "TIP")

    if gld_full is None:
        fail("Cannot download GLD (core series).")
    if GVZ is None:
        fail("Cannot download GVZ (signals depend on it).")

    GLD = gld_full["Close"].rename("GLD")
    df = pd.DataFrame({"GLD": GLD})
    df["GLD_HIGH"] = gld_full["High"]
    df["GLD_LOW"] = gld_full["Low"]
    for name, s in [("GVZ", GVZ), ("VIX", VIX), ("VIX3M", VIX3M), ("SLV", SLV),
                    ("DXY", DXY), ("TIP", TIP)]:
        if s is not None:
            df = df.join(s.rename(name), how="left")

    df = df.sort_index().loc[START_DATE:]
    df.index = pd.to_datetime(df.index)
    for col in ["GVZ", "VIX", "VIX3M", "DXY", "TIP"]:
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

    print("[5] Price-context indicators (EA parity, GLD proxy)...")
    df["EMA_FAST"] = df["GLD"].ewm(span=FAST_MA_PERIOD, adjust=False).mean()
    df["EMA_SLOW"] = df["GLD"].ewm(span=SLOW_MA_PERIOD, adjust=False).mean()
    df["RSI"] = wilder_rsi(df["GLD"], RSI_PERIOD)
    df["ATR"] = wilder_atr(df["GLD_HIGH"], df["GLD_LOW"], df["GLD"], ATR_PERIOD)
    df["DAY_RANGE"] = df["GLD_HIGH"] - df["GLD_LOW"]

    return df


# ------------------------- ENTRY STACK (mirrors the funded EA) -------------------------
def classify_trend(price, fma, sma):
    """Mirrors GetDualMATrend."""
    if fma <= 0 or sma <= 0 or np.isnan(fma) or np.isnan(sma):
        return TREND_NEUTRAL
    pct = (price - sma) / sma * 100.0
    f_above_s = fma > sma
    p_above_f = price > fma
    p_above_s = price > sma
    if f_above_s and p_above_f and pct > TREND_STRENGTH_MIN_PCT:
        return TREND_STRONG_UP
    if (not f_above_s) and (not p_above_f) and pct < -TREND_STRENGTH_MIN_PCT:
        return TREND_STRONG_DOWN
    if p_above_s and not f_above_s:
        return TREND_WEAK_UP
    if (not p_above_s) and f_above_s:
        return TREND_WEAK_DOWN
    return TREND_NEUTRAL


def rsi_min_for_trend(trend):
    """Mirrors GetRSIThresholdForLong."""
    if trend == TREND_STRONG_UP:
        return RSI_LONGMIN_STRONG_UP
    if trend == TREND_WEAK_UP:
        return RSI_LONGMIN_WEAK_UP
    if trend == TREND_NEUTRAL:
        return RSI_LONGMIN_NEUTRAL
    if trend == TREND_WEAK_DOWN:
        return RSI_LONGMIN_WEAK_DOWN
    return RSI_LONGMIN_DEFAULT


def readiness_score(valid):
    """Mirrors ComputeReadinessScore on the last SCORE_LOOKBACK settled rows."""
    tail = valid["zscore_ENSEMBLE"].iloc[-SCORE_LOOKBACK:]
    avg_ens = float(tail.mean())
    pers = float((tail > 0).mean())
    vrp = float(valid["zscore_VRP"].iloc[-1])
    reg = int(valid["regime_favorable"].iloc[-1])
    sc = 0.0
    if avg_ens >= SCORE_ENS_HIGH:
        sc += 1.0
    elif avg_ens >= SCORE_ENS_MID:
        sc += 0.5
    if vrp >= SCORE_VRP_HIGH:
        sc += 1.0
    elif vrp >= 0.0:
        sc += 0.5
    if pers >= SCORE_PERS_HIGH:
        sc += 1.0
    elif pers >= SCORE_PERS_MID:
        sc += 0.5
    if reg == 1:
        sc += 1.0
    return sc, avg_ens, pers


def decide_signal(valid):
    """Long-entry stack, gate for gate. Returns (signal, log_lines)."""
    log = []
    row = valid.iloc[-1]

    ens = float(row["zscore_ENSEMBLE"])
    z_vrp = float(row["zscore_VRP"])
    z_skew = float(row["zscore_SKEW"])
    z_pcr = float(row["zscore_PCR"])
    z_term = float(row["zscore_TERM"])
    regime_ok = int(row["regime_favorable"]) == 1
    macro_ok = int(row["macro_favorable"]) == 1

    # Gate 1: readiness score
    score, avg_ens, pers = readiness_score(valid)
    log.append(f"score gate: {score:.1f}/4 (avg_ens {avg_ens:+.2f}, pers {pers:.0%}) "
               f"need >= {MIN_DEPLOY_SCORE}")
    if score < MIN_DEPLOY_SCORE:
        return "NONE", log + ["BLOCKED by score gate"]

    # Gate 2: macro
    log.append(f"macro gate: macro_favorable={int(macro_ok)}")
    if not macro_ok:
        return "NONE", log + ["BLOCKED by macro gate"]

    # Gate 3: VRP gate (live config: OFF)
    if USE_VRP_GATE:
        log.append(f"vrp gate: z_VRP {z_vrp:+.2f} need > {VRP_MIN_Z}")
        if z_vrp <= VRP_MIN_Z:
            return "NONE", log + ["BLOCKED by VRP gate"]
    else:
        log.append("vrp gate: OFF (live funded config)")

    # Trend classification (EA parity, GLD proxy)
    price = float(row["GLD"])
    fma = float(row["EMA_FAST"])
    sma = float(row["EMA_SLOW"])
    trend = classify_trend(price, fma, sma)
    log.append(f"trend: {TREND_NAMES[trend]} (price {price:.2f}, ema50 {fma:.2f}, ema200 {sma:.2f})")

    # Regime + v19 override
    override_ok = (trend == TREND_STRONG_UP) or \
                  (REGIME_OVERRIDE_WEAKUP and trend == TREND_WEAK_UP)
    if not regime_ok and override_ok and ens >= REGIME_OVERRIDE_ENS:
        regime_ok = True
        log.append(f"regime: override fired (trend up, ens {ens:+.2f} >= {REGIME_OVERRIDE_ENS})")
    else:
        log.append(f"regime: favorable={int(regime_ok)}")

    # Trend filter: adjust ensemble and threshold
    filtered = ens
    long_thr = ENSEMBLE_LONG_THRESHOLD
    if trend == TREND_STRONG_UP:
        filtered = ens * 1.3 if ens > 0 else 0.0
    elif trend == TREND_STRONG_DOWN:
        filtered = 0.0 if ens > 0 else filtered
    elif trend == TREND_WEAK_UP:
        if ens < 0:
            filtered = 0.0
    elif trend == TREND_NEUTRAL:
        long_thr *= TRANSITION_MULT_NEUTRAL
    elif trend == TREND_WEAK_DOWN:
        long_thr *= TRANSITION_MULT_WEAKDN
    log.append(f"ensemble {ens:+.2f} -> filtered {filtered:+.2f}, long_thr {long_thr:.2f}")

    # RSI gate
    rsi = float(row["RSI"])
    rsi_min = rsi_min_for_trend(trend)
    log.append(f"rsi gate: RSI {rsi:.0f} need >= {rsi_min:.0f}")
    if filtered > 0 and rsi < rsi_min:
        return "NONE", log + ["BLOCKED by RSI gate"]

    # Flash-crash filter
    atr = float(row["ATR"])
    day_range = float(row["DAY_RANGE"])
    log.append(f"flash-crash: range {day_range:.2f} vs {FLASHCRASH_ATR_MULT} x ATR {atr:.2f}")
    if atr > 0 and day_range > atr * FLASHCRASH_ATR_MULT:
        return "NONE", log + ["BLOCKED by flash-crash filter"]

    # Regime requirement (post-override)
    if not regime_ok:
        return "NONE", log + ["BLOCKED by regime gate"]

    # Agreement: at least 1 individual directed signal >= 0.5
    agreeing = 0
    if filtered > 0:
        if z_vrp >= INDIV_SIGNAL_THRESHOLD:
            agreeing += 1
        if -z_skew >= INDIV_SIGNAL_THRESHOLD:
            agreeing += 1
        if -z_pcr >= INDIV_SIGNAL_THRESHOLD:
            agreeing += 1
        if z_term >= INDIV_SIGNAL_THRESHOLD:
            agreeing += 1
    log.append(f"agreement: {agreeing} signal(s) >= {INDIV_SIGNAL_THRESHOLD} "
               f"(need {MIN_SIGNALS_REQUIRED})")
    if agreeing < MIN_SIGNALS_REQUIRED:
        return "NONE", log + ["BLOCKED by agreement check"]

    # Final threshold
    if filtered == 0.0 or filtered < long_thr:
        return "NONE", log + [f"BLOCKED: filtered {filtered:+.2f} < threshold {long_thr:.2f}"]

    return "BUY", log + ["ALL GATES PASSED -> BUY"]


# ------------------------- FEED BUILD -------------------------
def build_feed_row(df):
    valid = df.dropna(subset=["zscore_ENSEMBLE"])
    if len(valid) < SCORE_LOOKBACK:
        fail("Not enough valid rows for the score gate.")

    row_date = valid.index[-1]
    age_days = (pd.Timestamp.today().normalize() - row_date.normalize()).days
    if age_days > 5:
        fail(f"Latest settled row is {age_days} days old ({row_date.date()}). Not publishing.")

    signal, log = decide_signal(valid)

    print("\n--- ENTRY STACK DIAGNOSTIC ---")
    for line in log:
        print("  " + line)
    print("------------------------------")

    row = valid.iloc[-1]
    ensemble = float(row["zscore_ENSEMBLE"])
    regime = int(row["regime_favorable"])
    macro = int(row["macro_favorable"])
    trend_er = float(row["trend_ER"]) if pd.notna(row["trend_ER"]) else 0.0

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = "schema,date,signal,ensemble,regime_favorable,macro_favorable,trend_er,generated_utc"
    line = (f"{SCHEMA_VERSION},{row_date.strftime('%Y.%m.%d')},{signal},"
            f"{ensemble:.4f},{regime},{macro},{trend_er:.4f},{generated}")
    return header + "\n" + line + "\n", signal, row_date, ensemble


def write_feed(content):
    os.makedirs(FEED_DIR, exist_ok=True)
    with open(FEED_FILE, "w", newline="\n") as f:
        f.write(content)
    print(f"[6] Feed written: {FEED_FILE}")


# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Gold Signal Feed Publisher v3 (EA-parity entry stack)")
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
