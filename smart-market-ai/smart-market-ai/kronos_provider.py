"""
Kronos forecast feature provider.

Wraps the vendored Kronos foundation model (shiyu-coder/Kronos, MIT) so its
forecasts can feed the smart-market feature set, mirroring the finnhub
fundamental-snapshot integration: static per-ticker scalars are joined into
the training and live feature frames.

Kronos is a pre-trained transformer trained on 12B+ historical K-line records.
Its training corpus likely overlaps the backtest window, so these features carry
a known data-snooping caveat: they are treated as signals, never as ground truth.

Runtime note: torch here is CPU-only. Each ticker forecast takes a few seconds
on the first run; results are cached in kronos_cache.json keyed by data hash so
retrains skip inference entirely.
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

CACHE_FILE = ROOT / "kronos_cache.json"


def load_env_file(path=ENV_FILE):
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


load_env_file()


def _env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


KRONOS_ENABLED = _env_flag("KRONOS_ENABLED")
KRONOS_MODEL = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small").strip()
KRONOS_TOKENIZER = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base").strip()
KRONOS_LOOKBACK = int(os.environ.get("KRONOS_LOOKBACK", "250"))
KRONOS_PRED_LEN = int(os.environ.get("KRONOS_PRED_LEN", "20"))
KRONOS_SAMPLES = int(os.environ.get("KRONOS_SAMPLES", "7"))
KRONOS_TEMPERATURE = float(os.environ.get("KRONOS_TEMPERATURE", "0.7"))
KRONOS_TOP_P = float(os.environ.get("KRONOS_TOP_P", "0.9"))
KRONOS_SEED = int(os.environ.get("KRONOS_SEED", "42"))

KRONOS_FEATURE_COLUMNS = [
    "kronos_exp_return",
    "kronos_vol_forecast",
    "kronos_trend",
    "kronos_up_prob",
]

_predictor = None
_models_failed = False


def load_json_file(path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json_file(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as exc:
        print(f"  Kronos: could not write cache {path}: {exc}")


def get_predictor():
    """Load the Kronos predictor once per process (None if disabled/unavailable)."""
    global _predictor, _models_failed
    if _predictor is not None or _models_failed:
        return _predictor
    if not KRONOS_ENABLED:
        return None
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER, revision=TOKENIZER_REVISION)
        model = Kronos.from_pretrained(KRONOS_MODEL, revision=MODEL_REVISION)
        tokenizer.eval()
        model.eval()
        _predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        print(f"  Kronos model loaded: {KRONOS_MODEL} (CPU)")
    except Exception as exc:
        print(f"  {chr(0x1B)}[33mKronos unavailable: {exc}{chr(0x1B)}[0m")
        _models_failed = True
        _predictor = None
    return _predictor


def _run_forecast(df):
    """Run a single Kronos forecast over the last KRONOS_LOOKBACK bars."""
    predictor = get_predictor()
    if predictor is None:
        return None

    tail = df.tail(KRONOS_LOOKBACK).copy()
    ohlcv = pd.DataFrame(index=tail.index)
    ohlcv["open"] = tail["open"].astype(float)
    ohlcv["high"] = tail["high"].astype(float)
    ohlcv["low"] = tail["low"].astype(float)
    ohlcv["close"] = tail["close"].astype(float)
    ohlcv["volume"] = tail["volume"].astype(float) if "volume" in tail.columns else 0.0
    ohlcv["amount"] = ohlcv["volume"] * ohlcv[["open", "high", "low", "close"]].mean(axis=1)
    ohlcv = ohlcv.dropna().astype(np.float32)
    if len(ohlcv) < KRONOS_LOOKBACK:
        return None

    x_timestamp = pd.Series(pd.to_datetime(ohlcv.index))
    y_timestamp = pd.Series(
        pd.bdate_range(start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), periods=KRONOS_PRED_LEN)
    )

    import torch

    torch.manual_seed(KRONOS_SEED)
    try:
        return predictor.predict(
            df=ohlcv.reset_index(drop=True),
            x_timestamp=x_timestamp.reset_index(drop=True),
            y_timestamp=y_timestamp,
            pred_len=KRONOS_PRED_LEN,
            T=KRONOS_TEMPERATURE,
            top_k=0,
            top_p=KRONOS_TOP_P,
            sample_count=KRONOS_SAMPLES,
            verbose=False,
        )
    except Exception as exc:
        print(f"  Kronos forecast failed: {exc}")
        return None


def _derive_features(pred_df, df):
    last_close = float(df["close"].iloc[-1])
    pred_close = pred_df["close"].astype(float).to_numpy()
    if len(pred_close) < 2 or not np.isfinite(last_close) or last_close <= 0:
        return {}

    horizon_return = pred_close[-1] / last_close - 1.0
    daily_rets = np.diff(np.log(np.clip(pred_close, 1e-9, None)))
    vol = float(np.std(daily_rets)) if len(daily_rets) else 0.0
    total_vol = vol * math.sqrt(len(pred_close)) + 1e-9
    trend = horizon_return / total_vol
    up_prob = 0.5 * (1.0 + math.erf(horizon_return / (total_vol * math.sqrt(2))))

    return {
        "kronos_exp_return": round(float(np.clip(horizon_return, -1, 1)), 6),
        "kronos_vol_forecast": round(float(np.clip(vol, 0, 1)), 6),
        "kronos_trend": round(float(np.clip(trend, -10, 10)), 6),
        "kronos_up_prob": round(float(np.clip(up_prob, 0, 1)), 6),
    }


def _data_hash(df):
    closes = df["close"].tail(KRONOS_LOOKBACK).round(6).astype(str)
    raw = "\n".join(closes.tolist())
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def kronos_forecast_features(ticker, df):
    """Return dict of kronos_* scalars for one ticker, or {} if unavailable.

    Mirrors fetch_fundamental_snapshot: a static per-ticker value set.
    """
    if not KRONOS_ENABLED:
        return {}
    if df is None or df.empty or "close" not in df.columns or len(df) < KRONOS_LOOKBACK:
        return {}

    cache = load_json_file(CACHE_FILE, {})
    key = f"{KRONOS_MODEL}|{KRONOS_LOOKBACK}|{KRONOS_PRED_LEN}|{KRONOS_SAMPLES}|{ticker}|{_data_hash(df)}"
    cached = cache.get(key)
    if cached and cached.get("features"):
        return dict(cached["features"])

    pred_df = _run_forecast(df)
    if pred_df is None or pred_df.empty:
        return {}
    features = _derive_features(pred_df, df)
    if not features:
        return {}

    cache[key] = {"time": time.time(), "last_date": str(df.index[-1]), "features": features}
    save_json_file(CACHE_FILE, cache)
    return features
