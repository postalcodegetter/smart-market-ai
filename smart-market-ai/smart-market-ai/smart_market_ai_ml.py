"""
SMART MARKET AI - Walk-Forward ML Predictor

Educational/informational only. Not financial advice.

Install:

"""

import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd



def require(pkg, install_name=None):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"Missing dependency: pip install {install_name or pkg}")
        sys.exit(1)


yf = require("yfinance")
colorama = require("colorama")
require("sklearn", "scikit-learn")
joblib = require("joblib")

from colorama import Fore, Style, init
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# Silence a benign sklearn informational warning about sklearn.utils.parallel
# config propagation. It fires in some joblib/sklearn version combos from
# estimators using n_jobs=-1; it does not affect results or model behaviour.
# The warning is emitted inside joblib workers, where module-level
# warnings.filters do not apply, so also seed a targeted PYTHONWARNINGS rule
# that loky-spawned worker processes pick up at interpreter startup.
if not os.environ.get("PYTHONWARNINGS"):
    os.environ["PYTHONWARNINGS"] = "ignore:`sklearn.utils.parallel.delayed`:UserWarning"
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*should be used",
    category=UserWarning,
)

try:
    from kronos_provider import KRONOS_FEATURE_COLUMNS, kronos_forecast_features
except Exception:
    KRONOS_FEATURE_COLUMNS = []
    kronos_forecast_features = lambda ticker, df: {}

try:
    import torch
except ImportError:
    torch = None


init(autoreset=True)

try:
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


ROOT = Path(__file__).parent
ENV_FILE = ROOT / ".env"
MEMORY_FILE = ROOT / "brain_memory_ml.json"
MODEL_FILE = ROOT / "market_model.joblib"
CHART_FILE = ROOT / "walk_forward_equity.png"
PAPER_TRADES_FILE = ROOT / "paper_trades.csv"
PATTERN_MEMORY_FILE = ROOT / "pattern_memory.json"
DB_FILE = ROOT / "smart_market_ai.db"
UNIVERSE_CACHE_FILE = ROOT / "dynamic_universe_cache.json"
FINNHUB_CACHE_FILE = ROOT / "finnhub_cache.json"

DEFAULT_TICKERS = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMD",
    "AVGO",
    "TSM",
    "ASML",
    "PLTR",
    "SMCI",
    "ARM",
    "CAT",
    "URI",
    "MLM",
    "VMC",
    "J",
    "ACM",
    "PWR",
    "FIX",
    "DHI",
    "LEN",
]
DEFAULT_LOOKBACK = "5y"
DEFAULT_HORIZON_DAYS = 20
DEFAULT_RETURN_THRESHOLD = 0.03
DEFAULT_TRAINING_MODE = "setups"
# Bias fix: these used to be asymmetric (0.54 up / 0.22 down), meaning a long
# only needed a 4-point edge over 50/50 while a short needed a 28-point edge.
# That structurally produced far more long signals than short signals
# regardless of what the model actually believed. Both thresholds now require
# the same distance from 0.50, so longs and shorts are held to an equal bar.
DEFAULT_PROB_UP = 0.58
DEFAULT_PROB_DOWN = 0.42
DEFAULT_MIN_COVERAGE = 8.0
# Meta-label gate guards: keep the gate from collapsing coverage to ~0 trades.
# The threshold is chosen to keep at least META_MIN_KEEP_FRACTION of traded rows,
# and if the gated result still falls below a floor vs the plain allocator, the
# gate is dropped entirely (see train_and_backtest).
META_MIN_KEEP_FRACTION = 0.30
META_MIN_KEEP_ROWS = 20
META_ABS_MIN_COVERAGE = 1.0
META_FLOOR_RELATIVE = 0.25
DEFAULT_FEE_BPS = 10
DEFAULT_STOP_LOSS = 0.06
DEFAULT_TAKE_PROFIT = 0.12
DEFAULT_ATR_STOP_MULT = 1.5
DEFAULT_ATR_TARGET_MULT = 3.0
MIN_ATR_STOP = 0.025
MAX_ATR_STOP = 0.10
DEFAULT_MAX_POSITION_SIZE = 0.15
DEFAULT_MAX_PORTFOLIO_EXPOSURE = 0.25
DEFAULT_MAX_SECTOR_EXPOSURE = 0.12
DEFAULT_DAILY_STOP = 0.008
DEFAULT_EQUITY_BREAKER = 0.12
DEFAULT_MAX_DAILY_TRADES = 3
DEFAULT_MAX_DAILY_SECTOR_TRADES = 1
DEFAULT_MAX_CONSECUTIVE_LOSS_DAYS = 3
DEFAULT_LOSS_COOLDOWN_DAYS = 5
DEFAULT_TARGET_TRADE_RISK = 0.02
DEFAULT_MIN_POSITION_SIZE = 0.02
DEFAULT_MAX_CORRELATION_PENALTY = 0.50
DEFAULT_BETA_LIMIT = 1.20
DEFAULT_TARGET_PORTFOLIO_VOL = 0.12
DEFAULT_MAX_PORTFOLIO_HEAT = 0.06
DEFAULT_FEATURE_DRIFT_Z_LIMIT = 3.0
DEFAULT_MIN_FINAL_TRADES = 25
DEFAULT_STRESS_SLIPPAGE_BPS = 25
DEFAULT_FINAL_TEST_FRACTION = 0.20
# Lopez de Prado (AFML Ch.7): purged walk-forward validation with embargo.
# Opt-in so existing runs keep the legacy TimeSeriesSplit behaviour; set
# PURGE_CV=1 to drop training rows whose label window overlaps each test fold.
PURGE_CV_ENABLED = os.environ.get("PURGE_CV", "0").strip().lower() not in ("0", "false", "no")
PURGE_EMBARGO_DAYS = int(os.environ.get("PURGE_EMBARGO_DAYS", "0"))
# PSR / Deflated Sharpe evaluation (ML4AM Ch.6). DEFAULT_SELECTION_TRIALS is
# the number of symmetric up/down threshold configs tried during tuning
# (C(14,2) = 91). MIN_DEPLOY_DSR gates deployability on beating the expected
# best of those trials.
DEFAULT_SELECTION_TRIALS = 91
MIN_DEPLOY_DSR = 0.5
# Fractional differentiation features (AFML Ch.5). Opt-in, default off.
ENABLE_FRACDIFF = os.environ.get("ENABLE_FRACDIFF", "0").strip().lower() not in ("0", "false", "no")
FRACDIFF_D = float(os.environ.get("FRACDIFF_D", "0.4"))
FRACDIFF_THRESHOLD = float(os.environ.get("FRACDIFF_THRESHOLD", "0.01"))
DEFAULT_UNIVERSE_SIZE = 20
DEFAULT_MIN_AVG_DOLLAR_VOLUME = 25_000_000
MIN_REGIME_MODEL_ROWS = 250
BROAD_ROBUSTNESS_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLI",
    "XLB",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLU",
    "SMH",
    "IGV",
    "ITB",
    "VNQ",
    "GLD",
    "TLT",
    "AAPL",
    "AMZN",
    "META",
    "ORCL",
    "CRM",
    "ADBE",
    "INTC",
    "MU",
    "QCOM",
    "AMAT",
    "LRCX",
    "DE",
    "NOC",
    "GE",
    "HON",
    "EMR",
    "ETN",
    "FCX",
    "NUE",
    "STLD",
    "PHM",
    "TOL",
    "HD",
    "LOW",
    "JPM",
    "BAC",
    "GS",
    "LLY",
    "UNH",
    "XOM",
    "CVX",
]
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_CACHE_TTL_SECONDS = 60 * 30
ENABLE_FINBERT = os.environ.get("ENABLE_FINBERT", "1").strip().lower() not in ("0", "false", "no")
FINBERT_MODEL = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert").strip()
FINBERT_PIPELINE = None
FINBERT_UNAVAILABLE = False
TOP_QUALITY_LEVELS = {"A", "B"}
ACTIONABLE_QUALITY_LEVELS = {"A", "B"}
NEAR_UP_BUFFER = 0.03
BENCHMARK_TICKERS = [
    "SPY",
    "QQQ",
    "^VIX",
    "SMH",
    "XLI",
    "XLB",
    "ITB",
    "RSP",
    "IWM",
    "HYG",
    "IEF",
]
MACRO_TICKERS = [
    "^TNX",
    "DX-Y.NYB",
    "CL=F",
    "GC=F",
]
SECTOR_MAP = {
    # Mega-cap tech / consumer tech
    "AAPL": "mega_cap_tech",
    "MSFT": "mega_cap_tech",
    "GOOGL": "mega_cap_tech",
    "AMZN": "consumer_discretionary",
    "META": "mega_cap_tech",
    "CRM": "software",
    "ORCL": "software",
    "ADBE": "software",
    "NFLX": "communication_services",
    # Semiconductors / AI infrastructure
    "NVDA": "semiconductors",
    "AMD": "semiconductors",
    "AVGO": "semiconductors",
    "TSM": "semiconductors",
    "ASML": "semiconductors",
    "ARM": "semiconductors",
    "INTC": "semiconductors",
    "MU": "semiconductors",
    "QCOM": "semiconductors",
    "AMAT": "semiconductors",
    "LRCX": "semiconductors",
    "SMCI": "ai_infrastructure",
    "PLTR": "software_ai",
    "CRWD": "software_ai",
    "PANW": "software_ai",
    "FTNT": "software_ai",
    "ZS": "software_ai",
    "NET": "software_ai",
    "SNOW": "software_ai",
    "MDB": "software_ai",
    # Financials
    "JPM": "banks",
    "GS": "banks",
    "MS": "banks",
    "BLK": "asset_management",
    "V": "payments",
    "MA": "payments",
    "BAC": "banks",
    "WFC": "banks",
    "SCHW": "banks",
    "AXP": "payments",
    "COIN": "fintech",
    # Healthcare
    "LLY": "healthcare",
    "UNH": "healthcare",
    "JNJ": "healthcare",
    "ABBV": "healthcare",
    "MRK": "healthcare",
    "TMO": "healthcare",
    "PFE": "healthcare",
    "BMY": "healthcare",
    # Consumer
    "WMT": "consumer_staples",
    "COST": "consumer_staples",
    "HD": "consumer_discretionary",
    "MCD": "consumer_discretionary",
    "KO": "consumer_staples",
    "PEP": "consumer_staples",
    "PG": "consumer_staples",
    "NKE": "consumer_discretionary",
    "SBUX": "consumer_discretionary",
    "TJX": "consumer_discretionary",
    # Industrials / aerospace / defense
    "CAT": "industrials",
    "DE": "industrials",
    "GE": "industrials",
    "RTX": "aerospace_defense",
    "HON": "industrials",
    "UPS": "industrials",
    "UNP": "industrials",
    "BA": "aerospace_defense",
    "LMT": "aerospace_defense",
    "GD": "aerospace_defense",
    "URI": "industrials",
    # Energy / materials
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "SLB": "energy",
    "OXY": "energy",
    "MLM": "materials",
    "VMC": "materials",
    # Homebuilders
    "DHI": "homebuilders",
    "LEN": "homebuilders",
    "PHM": "homebuilders",
    "TOL": "homebuilders",
    # Auto / mobility / travel
    "TSLA": "auto",
    "F": "auto",
    "GM": "auto",
    "UBER": "rideshare",
    "ABNB": "travel",
    # Engineering / construction
    "J": "engineering_construction",
    "ACM": "engineering_construction",
    "PWR": "engineering_construction",
    "FIX": "engineering_construction",
    # ETFs / asset-class buckets
    "SPY": "broad_market",
    "QQQ": "broad_market",
    "IWM": "broad_market",
    "RSP": "broad_market",
    "XLK": "sector_etf",
    "XLE": "sector_etf",
    "XLF": "sector_etf",
    "SMH": "sector_etf",
    "GLD": "commodities",
    "SLV": "commodities",
    "USO": "commodities",
    "TLT": "rates",
    "HYG": "credit",
}
SCREENER_NAMES = [
    "most_actives",
    "day_gainers",
    "day_losers",
    "undervalued_growth_stocks",
    "growth_technology_stocks",
    "aggressive_small_caps",
]
POSITIVE_NEWS_WORDS = {
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "raises",
    "raised",
    "growth",
    "record",
    "strong",
    "surge",
    "rally",
    "bullish",
    "profit",
    "profits",
    "wins",
    "contract",
}
NEGATIVE_NEWS_WORDS = {
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "cuts",
    "cut",
    "weak",
    "probe",
    "lawsuit",
    "falls",
    "fall",
    "slump",
    "bearish",
    "loss",
    "losses",
    "warning",
    "layoffs",
}
NEWS_CATEGORY_RULES = [
    ("earnings_beat", 1.5, ["beats estimates", "earnings beat", "beat expectations", "tops estimates", "better-than-expected"]),
    ("earnings_miss", -1.5, ["misses estimates", "earnings miss", "missed expectations", "falls short", "worse-than-expected"]),
    ("guidance_raise", 1.4, ["raises guidance", "raises outlook", "boosts forecast", "lifts forecast", "higher guidance"]),
    ("guidance_cut", -1.6, ["cuts guidance", "lowers outlook", "slashes forecast", "weak guidance", "reduced guidance"]),
    ("analyst_upgrade", 1.0, ["upgraded", "upgrade", "price target raised", "raises price target", "initiated at buy"]),
    ("analyst_downgrade", -1.0, ["downgraded", "downgrade", "price target cut", "cuts price target", "initiated at sell"]),
    ("contract_win", 1.0, ["wins contract", "awarded contract", "new contract", "partnership", "deal with"]),
    ("legal_regulatory", -1.4, ["lawsuit", "probe", "investigation", "sec", "doj", "antitrust", "regulator", "recall"]),
    ("layoffs_cost_cuts", -0.8, ["layoffs", "job cuts", "cuts jobs", "restructuring", "cost cuts"]),
    ("ai_capex_concern", -1.1, ["ai spending", "capex concern", "capital spending", "ai bubble", "valuation concern"]),
    ("valuation_risk", -0.8, ["overvalued", "valuation", "bubble", "too expensive", "multiple compression"]),
    ("product_strength", 0.8, ["launches", "new product", "demand strong", "orders surge", "backlog grows"]),
    ("margin_pressure", -1.0, ["margin pressure", "margin squeeze", "gross margin falls", "profit warning"]),
]
PRICE_ACTION_WORDS = {
    "up": ["shares rise", "stock rises", "shares jump", "stock jumps", "rallies", "surges", "gains"],
    "down": ["shares fall", "stock falls", "shares drop", "stock drops", "slumps", "tumbles", "slides"],
}

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "volatility_10d",
    "volatility_20d",
    "atr_pct",
    "volume_z_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct_b",
    "stoch_14",
    "adx_dir",
    "obv_10d",
    "vwap_dist",
    "sma_20_dist",
    "sma_50_dist",
    "sma_200_dist",
    "ema_20_50",
    "range_pct",
    "close_position",
    "dollar_volume_z_20d",
    "volume_price_pressure",
    "obv_accel",
    "sweep_low_20",
    "sweep_high_20",
    "failed_breakout_20",
    "failed_breakdown_20",
    "breakout_volume_20",
    "breakdown_volume_20",
    "vwap_reclaim",
    "vwap_rejection",
    "volatility_expansion",
    "gap_pct",
    "relative_volume_20d",
    "effort_vs_result",
    "absorption_up",
    "absorption_down",
    "upper_wick_pct",
    "lower_wick_pct",
    "wick_reversal_up",
    "wick_reversal_down",
    "chaikin_money_flow_20",
    "money_flow_index_14",
    "accumulation_distribution_10d",
    "pocket_pivot",
    "volume_dry_up",
    "range_expansion_up",
    "range_expansion_down",
    "quiet_accumulation",
    "quiet_distribution",
    "peer_ret_5d",
    "peer_ret_20d",
    "relative_strength_20d",
    "whale_footprint_score",
    "tnx_level",
    "tnx_change_5d",
    "tnx_trend_50",
    "dxy_trend_50",
    "dxy_ret_20",
    "oil_ret_20",
    "gold_ret_20",
    "fund_rev_growth",
    "fund_eps_growth",
    "fund_profit_margin",
    "fund_debt_equity",
    "fund_forward_pe",
    "fund_rec_score",
] + KRONOS_FEATURE_COLUMNS + (["frac_diff_close", "frac_diff_volume"] if ENABLE_FRACDIFF else [])

# Phase 1 cleanup: these features are retained in the raw dataset for audits/diagnostics,
# but removed from model training and live inference because the redundancy audit showed
# near-duplicate information.
REDUNDANT_FEATURES_TO_DROP = [
    "failed_breakdown_20",
    "failed_breakout_20",
    "macd_signal",
    "dollar_volume_z_20d",
    "sma_20_dist",
    "stoch_14",
    "adx_dir",
]

ACTIVE_FEATURE_COLUMNS = [
    col for col in FEATURE_COLUMNS
    if col not in REDUNDANT_FEATURES_TO_DROP
]


def model_feature_columns():
    return ACTIVE_FEATURE_COLUMNS


def section(title):
    print(f"\n{Fore.CYAN}{'-' * 72}")
    print(f"  {title}")
    print(f"{'-' * 72}{Style.RESET_ALL}")


def load_memory():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created": dt.datetime.now().isoformat(),
        "last_run": None,
        "runs": [],
        "best_model": None,
    }


def save_memory(mem):
    mem["last_run"] = dt.datetime.now().isoformat()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)
    print(f"  {Fore.GREEN}Memory saved -> {MEMORY_FILE}{Style.RESET_ALL}")


def log_model_run_to_db(run, tickers):
    db_insert(
        "model_runs",
        {
            "trained_at": run.get("trained_at"),
            "model_name": run.get("name"),
            "accuracy": run.get("accuracy"),
            "coverage": run.get("coverage"),
            "profit_factor": str(run.get("profit_factor")),
            "portfolio_profit_factor": str(run.get("portfolio_profit_factor")),
            "avg_return": run.get("avg_return"),
            "max_drawdown": run.get("max_drawdown"),
            "brier": run.get("brier"),
            "sharpe": run.get("sharpe"),
            "psr": run.get("psr"),
            "dsr": run.get("dsr"),
            "final_accuracy": run.get("final_accuracy"),
            "final_coverage": run.get("final_coverage"),
            "final_profit_factor": run.get("final_profit_factor"),
            "final_portfolio_profit_factor": run.get("final_portfolio_profit_factor"),
            "final_max_drawdown": run.get("final_max_drawdown"),
            "prob_up": run.get("prob_up"),
            "prob_down": run.get("prob_down"),
            "stop_loss": run.get("stop_loss"),
            "take_profit": run.get("take_profit"),
            "max_position_size": run.get("max_position_size"),
            "max_portfolio_exposure": run.get("max_portfolio_exposure"),
            "max_sector_exposure": run.get("max_sector_exposure"),
            "daily_stop": run.get("daily_stop"),
            "equity_breaker": run.get("equity_breaker"),
            "training_mode": run.get("training_mode"),
            "tickers": ",".join(tickers),
        },
    )


def log_live_predictions_to_db(records):
    run_at = dt.datetime.now().isoformat()
    for record in records:
        db_insert(
            "live_predictions",
            {
                "run_at": run_at,
                "ticker": record.get("ticker"),
                "price": record.get("price"),
                "signal": record.get("signal"),
                "prob_up": record.get("prob_up"),
                "quality": record.get("quality"),
                "position_size": record.get("position_size"),
                "rsi": record.get("rsi"),
                "whale_score": record.get("whale_score"),
                "risk_on_score": record.get("risk_on_score"),
                "fundamental_score": record.get("fundamental_score"),
                "news_score": record.get("news_score"),
                "event_score": record.get("event_score"),
                "intelligence_score": record.get("intelligence_score"),
                "event_note": record.get("event_note"),
                "provider": record.get("intelligence_provider"),
                "explanation": explain_record(record),
                "horizon_days": record.get("horizon_days"),
                "model_trained_at": record.get("model_trained_at"),
            },
        )


def log_paper_trade_to_db(row):
    db_insert(
        "paper_trades",
        {
            "saved_at": row.get("saved_at"),
            "signal_date": row.get("signal_date"),
            "target_date": row.get("target_date"),
            "ticker": row.get("ticker"),
            "entry_price": row.get("entry_price"),
            "signal": row.get("signal"),
            "prob_up": row.get("prob_up"),
            "quality": row.get("quality"),
            "position_size": row.get("position_size"),
            "whale_score": row.get("whale_score"),
            "risk_on_score": row.get("risk_on_score"),
            "rsi": row.get("rsi"),
            "fundamental_score": row.get("fundamental_score"),
            "news_score": row.get("news_score"),
            "event_score": row.get("event_score"),
            "intelligence_score": row.get("intelligence_score"),
            "event_note": row.get("event_note"),
            "provider": row.get("intelligence_provider"),
            "stop_loss": row.get("stop_loss"),
            "take_profit": row.get("take_profit"),
            "horizon_days": row.get("horizon_days"),
            "status": row.get("status", "OPEN"),
            "checked_at": row.get("checked_at"),
            "exit_date": row.get("exit_date"),
            "exit_price": row.get("exit_price"),
            "actual_return": row.get("actual_return"),
            "correct": row.get("correct"),
        },
    )


def log_pattern_to_db(record):
    db_insert(
        "pattern_memory",
        {
            "closed_at": record.get("closed_at"),
            "ticker": record.get("ticker"),
            "signal_date": record.get("signal_date"),
            "target_date": record.get("target_date"),
            "signal": record.get("signal"),
            "quality": record.get("quality"),
            "setup": record.get("setup"),
            "tags": record.get("tags"),
            "prob_up": record.get("prob_up"),
            "whale_score": record.get("whale_score"),
            "risk_on_score": record.get("risk_on_score"),
            "rsi": record.get("rsi"),
            "news_score": record.get("news_score"),
            "intelligence_score": record.get("intelligence_score"),
            "event_note": record.get("event_note"),
            "result": record.get("result"),
            "correct": record.get("correct"),
        },
    )


def log_robustness_to_db(row):
    db_insert(
        "robustness_results",
        {
            "run_at": dt.datetime.now().isoformat(),
            "horizon": row.get("horizon"),
            "move_threshold": row.get("move"),
            "stop_loss": row.get("stop"),
            "take_profit": row.get("target"),
            "risk_mode": row.get("risk_mode"),
            "training_mode": row.get("training_mode"),
            "accuracy": row.get("accuracy"),
            "coverage": row.get("coverage"),
            "avg_return": row.get("avg_return"),
            "profit_factor": str(row.get("profit_factor")),
            "portfolio_profit_factor": str(row.get("portfolio_profit_factor")),
            "max_drawdown": row.get("max_drawdown"),
            "final_accuracy": row.get("final_accuracy"),
            "final_coverage": row.get("final_coverage"),
            "final_profit_factor": row.get("final_profit_factor"),
            "final_portfolio_profit_factor": row.get("final_portfolio_profit_factor"),
            "final_max_drawdown": row.get("final_max_drawdown"),
            "target_first_profit_factor": row.get("target_first_profit_factor"),
            "stress_profit_factor": row.get("stress_profit_factor"),
            "stress_slippage_bps": row.get("stress_slippage_bps"),
            "down_threshold": row.get("down_threshold"),
            "up_threshold": row.get("up_threshold"),
            "strict_filter": row.get("strict_filter"),
            "max_position_size": row.get("max_position_size"),
            "max_portfolio_exposure": row.get("max_portfolio_exposure"),
            "max_sector_exposure": row.get("max_sector_exposure"),
            "daily_stop": row.get("daily_stop"),
            "equity_breaker": row.get("equity_breaker"),
            "score": row.get("score"),
        },
    )


def load_json_file(path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS live_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT,
                ticker TEXT,
                price REAL,
                signal TEXT,
                prob_up REAL,
                quality TEXT,
                position_size REAL,
                rsi REAL,
                whale_score REAL,
                risk_on_score REAL,
                fundamental_score REAL,
                news_score REAL,
                event_score REAL,
                intelligence_score REAL,
                event_note TEXT,
                provider TEXT,
                explanation TEXT,
                horizon_days INTEGER,
                model_trained_at TEXT
            );

            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT,
                model_name TEXT,
                accuracy REAL,
                coverage REAL,
                profit_factor TEXT,
                portfolio_profit_factor TEXT,
                avg_return REAL,
                max_drawdown REAL,
                brier REAL,
                final_accuracy REAL,
                final_coverage REAL,
                final_profit_factor TEXT,
                final_portfolio_profit_factor TEXT,
                final_max_drawdown REAL,
                prob_up REAL,
                prob_down REAL,
                stop_loss REAL,
                take_profit REAL,
                max_position_size REAL,
                max_portfolio_exposure REAL,
                max_sector_exposure REAL,
                daily_stop REAL,
                equity_breaker REAL,
                training_mode TEXT,
                tickers TEXT
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at TEXT,
                signal_date TEXT,
                target_date TEXT,
                ticker TEXT,
                entry_price REAL,
                signal TEXT,
                prob_up REAL,
                quality TEXT,
                position_size REAL,
                whale_score REAL,
                risk_on_score REAL,
                rsi REAL,
                fundamental_score REAL,
                news_score REAL,
                event_score REAL,
                intelligence_score REAL,
                event_note TEXT,
                provider TEXT,
                stop_loss REAL,
                take_profit REAL,
                horizon_days INTEGER,
                status TEXT,
                checked_at TEXT,
                exit_date TEXT,
                exit_price REAL,
                actual_return REAL,
                correct INTEGER
            );

            CREATE TABLE IF NOT EXISTS pattern_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                closed_at TEXT,
                ticker TEXT,
                signal_date TEXT,
                target_date TEXT,
                signal TEXT,
                quality TEXT,
                setup TEXT,
                tags TEXT,
                prob_up REAL,
                whale_score REAL,
                risk_on_score REAL,
                rsi REAL,
                news_score REAL,
                intelligence_score REAL,
                event_note TEXT,
                result REAL,
                correct INTEGER
            );

            CREATE TABLE IF NOT EXISTS robustness_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT,
                horizon INTEGER,
                move_threshold REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_mode TEXT,
                training_mode TEXT,
                accuracy REAL,
                coverage REAL,
                avg_return REAL,
                profit_factor TEXT,
                portfolio_profit_factor TEXT,
                max_drawdown REAL,
                final_accuracy REAL,
                final_coverage REAL,
                final_profit_factor TEXT,
                final_portfolio_profit_factor TEXT,
                final_max_drawdown REAL,
                target_first_profit_factor TEXT,
                stress_profit_factor TEXT,
                stress_slippage_bps REAL,
                down_threshold REAL,
                up_threshold REAL,
                strict_filter INTEGER,
                max_position_size REAL,
                max_portfolio_exposure REAL,
                max_sector_exposure REAL,
                daily_stop REAL,
                equity_breaker REAL,
                score REAL
            );
            """
        )
        ensure_db_columns(conn)


def ensure_db_columns(conn):
    additions = {
        "model_runs": {
            "portfolio_profit_factor": "TEXT",
            "final_accuracy": "REAL",
            "final_coverage": "REAL",
            "final_profit_factor": "TEXT",
            "final_portfolio_profit_factor": "TEXT",
            "final_max_drawdown": "REAL",
            "max_sector_exposure": "REAL",
            "daily_stop": "REAL",
            "equity_breaker": "REAL",
            "training_mode": "TEXT",
            "sharpe": "REAL",
            "psr": "REAL",
            "dsr": "REAL",
        },
        "robustness_results": {
            "portfolio_profit_factor": "TEXT",
            "final_accuracy": "REAL",
            "final_coverage": "REAL",
            "final_profit_factor": "TEXT",
            "final_portfolio_profit_factor": "TEXT",
            "final_max_drawdown": "REAL",
            "target_first_profit_factor": "TEXT",
            "stress_profit_factor": "TEXT",
            "stress_slippage_bps": "REAL",
            "risk_mode": "TEXT",
            "training_mode": "TEXT",
            "max_sector_exposure": "REAL",
            "daily_stop": "REAL",
            "equity_breaker": "REAL",
        },
    }
    for table, columns in additions.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, col_type in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def db_insert(table, row):
    init_db()
    clean = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            clean[key] = json.dumps(value)
        elif isinstance(value, (np.bool_, bool)):
            clean[key] = int(value)
        else:
            try:
                clean[key] = None if pd.isna(value) else value
            except Exception:
                clean[key] = value
    columns = ", ".join(clean.keys())
    placeholders = ", ".join(["?"] * len(clean))
    with db_connect() as conn:
        conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", list(clean.values()))


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
    except Exception as exc:
        print(f"  {Fore.YELLOW}Could not load .env file: {exc}{Style.RESET_ALL}")


def finnhub_api_key():
    return os.environ.get("FINNHUB_API_KEY", "").strip()


def finnhub_get(endpoint, params=None, ttl_seconds=FINNHUB_CACHE_TTL_SECONDS):
    key = finnhub_api_key()
    if not key:
        return None

    params = params or {}
    params["token"] = key
    cache_key = endpoint + "?" + urllib.parse.urlencode(sorted(params.items()))
    cache = load_json_file(FINNHUB_CACHE_FILE, {})
    cached = cache.get(cache_key)
    now = time.time()
    if cached and now - cached.get("time", 0) < ttl_seconds:
        return cached.get("data")

    url = FINNHUB_BASE_URL + endpoint + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if cached:
            return cached.get("data")
        print(f"  {Fore.YELLOW}Finnhub unavailable for {endpoint}: {exc}{Style.RESET_ALL}")
        return None

    cache[cache_key] = {"time": now, "data": data}
    save_json_file(FINNHUB_CACHE_FILE, cache)
    return data


def get_finbert_pipeline():
    global FINBERT_PIPELINE, FINBERT_UNAVAILABLE
    if not ENABLE_FINBERT or FINBERT_UNAVAILABLE:
        return None
    if FINBERT_PIPELINE is not None:
        return FINBERT_PIPELINE
    try:
        from transformers import pipeline

        FINBERT_PIPELINE = pipeline("text-classification", model=FINBERT_MODEL, tokenizer=FINBERT_MODEL)
        return FINBERT_PIPELINE
    except Exception as exc:
        FINBERT_UNAVAILABLE = True
        print(f"  {Fore.YELLOW}FinBERT unavailable, using rule-based news scoring: {exc}{Style.RESET_ALL}")
        return None


def finbert_score_titles(titles):
    pipe = get_finbert_pipeline()
    if pipe is None or not titles:
        return []
    try:
        outputs = pipe(titles, truncation=True)
    except Exception as exc:
        print(f"  {Fore.YELLOW}FinBERT scoring failed, using rule-based news scoring: {exc}{Style.RESET_ALL}")
        return []

    results = []
    for output in outputs:
        label = str(output.get("label", "")).lower()
        confidence = safe_float(output.get("score"), 0)
        if "positive" in label:
            signed = confidence
            tag = "finbert_positive"
        elif "negative" in label:
            signed = -confidence
            tag = "finbert_negative"
        else:
            signed = 0
            tag = "finbert_neutral"
        results.append((signed, tag, confidence))
    return results


def fetch(ticker, period=DEFAULT_LOOKBACK, interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def clean_symbol(symbol):
    if not symbol:
        return None
    symbol = str(symbol).upper().strip()
    if not symbol or "." in symbol or "^" in symbol or "/" in symbol:
        return None
    blocked_suffixes = ("-W", "-WT", "-U", "-UN", "-P")
    if any(symbol.endswith(suffix) for suffix in blocked_suffixes):
        return None
    return symbol


def symbols_from_screener_response(response):
    found = []

    def walk(value):
        if isinstance(value, dict):
            symbol = value.get("symbol")
            cleaned = clean_symbol(symbol)
            if cleaned:
                found.append(cleaned)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(response)
    return found


def load_cached_universe():
    if not UNIVERSE_CACHE_FILE.exists():
        return []
    try:
        with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        return [ticker for ticker in tickers if clean_symbol(ticker)]
    except Exception:
        return []


def save_cached_universe(tickers, source):
    payload = {
        "created": dt.datetime.now().isoformat(),
        "source": source,
        "tickers": tickers,
    }
    with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def screener_candidates(limit=100):
    candidates = []
    if not hasattr(yf, "screen"):
        return candidates

    for screen_name in SCREENER_NAMES:
        try:
            response = yf.screen(screen_name, count=limit)
            candidates.extend(symbols_from_screener_response(response))
        except Exception as exc:
            print(f"  {Fore.YELLOW}Screener skipped: {screen_name} ({exc}){Style.RESET_ALL}")

    seen = set()
    unique = []
    for ticker in candidates:
        if ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    return unique


def validate_universe_candidates(candidates, universe_size, period="2y"):
    rows = []
    for ticker in candidates:
        if len(rows) >= universe_size * 3:
            break
        df = fetch(ticker, period=period)
        if len(df) < 180:
            continue
        avg_dollar_volume = float((df["Close"] * df["Volume"]).tail(60).mean())
        last_price = float(df["Close"].iloc[-1])
        if avg_dollar_volume < DEFAULT_MIN_AVG_DOLLAR_VOLUME or last_price < 5:
            continue
        momentum_20 = float(df["Close"].pct_change(20).iloc[-1])
        volatility_20 = float(df["Close"].pct_change().rolling(20).std().iloc[-1])
        rows.append(
            {
                "ticker": ticker,
                "avg_dollar_volume": avg_dollar_volume,
                "momentum_20": momentum_20,
                "volatility_20": volatility_20,
            }
        )

    if not rows:
        return []

    data = pd.DataFrame(rows)
    data["liquidity_rank"] = data["avg_dollar_volume"].rank(pct=True)
    data["momentum_rank"] = data["momentum_20"].abs().rank(pct=True)
    data["volatility_rank"] = data["volatility_20"].rank(pct=True)
    data["score"] = data["liquidity_rank"] * 0.55 + data["momentum_rank"] * 0.25 + data["volatility_rank"] * 0.20
    return data.sort_values("score", ascending=False)["ticker"].head(universe_size).tolist()


def build_dynamic_universe(universe_size=DEFAULT_UNIVERSE_SIZE):
    section("DYNAMIC UNIVERSE SCREENER")
    print("  Pulling candidates from yfinance screeners, then filtering for liquidity and enough history.")
    candidates = screener_candidates(limit=100)
    if not candidates:
        cached = load_cached_universe()
        if cached:
            print(f"  {Fore.YELLOW}Screener unavailable. Using cached universe.{Style.RESET_ALL}")
            return cached[:universe_size]
        print(f"  {Fore.YELLOW}Screener unavailable. Falling back to default tickers.{Style.RESET_ALL}")
        return DEFAULT_TICKERS[:universe_size]

    print(f"  Raw candidates: {len(candidates)}")
    tickers = validate_universe_candidates(candidates, universe_size)
    if not tickers:
        cached = load_cached_universe()
        if cached:
            print(f"  {Fore.YELLOW}No validated candidates. Using cached universe.{Style.RESET_ALL}")
            return cached[:universe_size]
        print(f"  {Fore.YELLOW}No validated candidates. Falling back to default tickers.{Style.RESET_ALL}")
        return DEFAULT_TICKERS[:universe_size]

    save_cached_universe(tickers, "yfinance_screeners")
    print(f"  {Fore.GREEN}Dynamic universe saved -> {UNIVERSE_CACHE_FILE}{Style.RESET_ALL}")
    print(f"  Using: {Fore.YELLOW}{', '.join(tickers)}{Style.RESET_ALL}")
    return tickers


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bool_as_float(series):
    return series.astype(float)


def frac_diff(series, d=FRACDIFF_D, thresh=FRACDIFF_THRESHOLD):
    """Fractionally differentiate a series (Lopez de Prado, AFML Ch.5).

    Uses the expanding-window weighted-sum approximation: weights decay as
    w_k = w_{k-1} * (d - k + 1) / k and are truncated below `thresh`. Lower d
    keeps more memory but is less stationary; d in [0, 1] is the standard range.
    """
    series = series.astype(float).fillna(method="ffill").fillna(0.0)
    n = len(series)
    if n == 0:
        return pd.Series(dtype=float, index=series.index)
    weights = [1.0]
    for k in range(1, n):
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        if abs(w) < thresh:
            break
    weights = np.asarray(weights)
    out = np.full(n, np.nan)
    for i in range(len(weights), n + 1):
        out[i - 1] = float(np.dot(weights, series.iloc[i - len(weights):i]))
    return pd.Series(out, index=series.index)


def make_features(df):
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"].replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    returns = c.pct_change()
    out["ret_1d"] = returns
    out["ret_3d"] = c.pct_change(3)
    out["ret_5d"] = c.pct_change(5)
    out["ret_10d"] = c.pct_change(10)
    out["ret_20d"] = c.pct_change(20)
    out["volatility_10d"] = returns.rolling(10).std()
    out["volatility_20d"] = returns.rolling(20).std()
    out["volume_z_20d"] = (v - v.rolling(20).mean()) / v.rolling(20).std()

    out["rsi_14"] = rsi(c)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    out["bb_pct_b"] = (c - (sma20 - 2 * std20)) / (4 * std20)

    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    out["stoch_14"] = 100 * (c - lo14) / (hi14 - lo14)

    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(span=14, adjust=False).mean() / atr
    out["adx_dir"] = (pdi - ndi) / 100
    out["atr_pct"] = atr / c

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    out["obv_10d"] = obv.pct_change(10).replace([np.inf, -np.inf], np.nan)

    typ = (h + l + c) / 3
    vwap = (typ * v).cumsum() / v.cumsum()
    out["vwap_dist"] = c / vwap - 1

    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    out["sma_20_dist"] = c / sma20 - 1
    out["sma_50_dist"] = c / sma50 - 1
    out["sma_200_dist"] = c / sma200 - 1
    out["ema_20_50"] = ema20 / ema50 - 1

    out["range_pct"] = (h - l) / c
    out["close_position"] = (c - l) / (h - l)
    dollar_volume = c * v
    out["dollar_volume_z_20d"] = (dollar_volume - dollar_volume.rolling(20).mean()) / dollar_volume.rolling(20).std()
    out["volume_price_pressure"] = out["ret_1d"] * out["volume_z_20d"]
    out["obv_accel"] = out["obv_10d"] - out["obv_10d"].rolling(10).mean()

    prior_high_20 = h.rolling(20).max().shift(1)
    prior_low_20 = l.rolling(20).min().shift(1)
    high_break = h > prior_high_20
    low_break = l < prior_low_20
    close_back_under_high = c < prior_high_20
    close_back_over_low = c > prior_low_20
    volume_spike = out["volume_z_20d"] > 1.5

    out["sweep_low_20"] = bool_as_float(low_break & close_back_over_low)
    out["sweep_high_20"] = bool_as_float(high_break & close_back_under_high)
    out["failed_breakout_20"] = bool_as_float(high_break & (c < prior_high_20))
    out["failed_breakdown_20"] = bool_as_float(low_break & (c > prior_low_20))
    out["breakout_volume_20"] = bool_as_float((c > prior_high_20) & volume_spike & (out["close_position"] > 0.70))
    out["breakdown_volume_20"] = bool_as_float((c < prior_low_20) & volume_spike & (out["close_position"] < 0.30))
    out["vwap_reclaim"] = bool_as_float((c > vwap) & (c.shift(1) <= vwap.shift(1)) & volume_spike)
    out["vwap_rejection"] = bool_as_float((c < vwap) & (c.shift(1) >= vwap.shift(1)) & volume_spike)
    out["volatility_expansion"] = out["volatility_10d"] / out["volatility_20d"]
    out["gap_pct"] = df["Open"] / c.shift(1) - 1

    avg_volume_20 = v.rolling(20).mean()
    out["relative_volume_20d"] = v / avg_volume_20
    out["effort_vs_result"] = out["relative_volume_20d"] / (out["ret_1d"].abs() * 100 + 0.25)
    out["absorption_up"] = bool_as_float(
        (out["relative_volume_20d"] > 1.8)
        & (out["range_pct"] < out["range_pct"].rolling(20).median())
        & (out["close_position"] > 0.60)
    )
    out["absorption_down"] = bool_as_float(
        (out["relative_volume_20d"] > 1.8)
        & (out["range_pct"] < out["range_pct"].rolling(20).median())
        & (out["close_position"] < 0.40)
    )

    candle_range = (h - l).replace(0, np.nan)
    body_high = pd.concat([df["Open"], c], axis=1).max(axis=1)
    body_low = pd.concat([df["Open"], c], axis=1).min(axis=1)
    out["upper_wick_pct"] = (h - body_high) / candle_range
    out["lower_wick_pct"] = (body_low - l) / candle_range
    out["wick_reversal_up"] = bool_as_float(
        (out["lower_wick_pct"] > 0.45)
        & (out["close_position"] > 0.60)
        & (out["relative_volume_20d"] > 1.2)
    )
    out["wick_reversal_down"] = bool_as_float(
        (out["upper_wick_pct"] > 0.45)
        & (out["close_position"] < 0.40)
        & (out["relative_volume_20d"] > 1.2)
    )

    money_flow_multiplier = ((c - l) - (h - c)) / candle_range
    money_flow_volume = money_flow_multiplier * v
    out["chaikin_money_flow_20"] = money_flow_volume.rolling(20).sum() / v.rolling(20).sum()

    typical_price = (h + l + c) / 3
    raw_money_flow = typical_price * v
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    money_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum().replace(0, np.nan)
    out["money_flow_index_14"] = 100 - 100 / (1 + money_ratio)

    accumulation_distribution = money_flow_volume.fillna(0).cumsum()
    out["accumulation_distribution_10d"] = accumulation_distribution.pct_change(10).replace([np.inf, -np.inf], np.nan)
    down_volume_10 = v.where(c < c.shift(1), 0).rolling(10).max()
    out["pocket_pivot"] = bool_as_float((c > c.shift(1)) & (v > down_volume_10) & (out["close_position"] > 0.65))
    out["volume_dry_up"] = bool_as_float((out["relative_volume_20d"] < 0.55) & (out["range_pct"] < out["range_pct"].rolling(20).median()))
    out["range_expansion_up"] = bool_as_float(
        (out["range_pct"] > out["range_pct"].rolling(20).mean() * 1.5)
        & (out["close_position"] > 0.75)
        & (out["relative_volume_20d"] > 1.3)
    )
    out["range_expansion_down"] = bool_as_float(
        (out["range_pct"] > out["range_pct"].rolling(20).mean() * 1.5)
        & (out["close_position"] < 0.25)
        & (out["relative_volume_20d"] > 1.3)
    )
    out["quiet_accumulation"] = bool_as_float(
        (out["volume_dry_up"] > 0)
        & (out["chaikin_money_flow_20"] > 0.05)
        & (out["sma_20_dist"] > -0.03)
    )
    out["quiet_distribution"] = bool_as_float(
        (out["volume_dry_up"] > 0)
        & (out["chaikin_money_flow_20"] < -0.05)
        & (out["sma_20_dist"] < 0.03)
    )
    if ENABLE_FRACDIFF:
        log_close = np.log(c.replace(0, np.nan))
        out["frac_diff_close"] = frac_diff(log_close, d=FRACDIFF_D, thresh=FRACDIFF_THRESHOLD)
        out["frac_diff_volume"] = frac_diff(v.fillna(0), d=FRACDIFF_D, thresh=FRACDIFF_THRESHOLD)
    return out.replace([np.inf, -np.inf], np.nan)


def make_benchmark_features(period):
    frames = []
    names = {
        "SPY": "spy",
        "QQQ": "qqq",
        "^VIX": "vix",
        "SMH": "smh",
        "XLI": "xli",
        "XLB": "xlb",
        "ITB": "itb",
        "RSP": "rsp",
        "IWM": "iwm",
        "HYG": "hyg",
        "IEF": "ief",
        "^TNX": "tnx",
        "DX-Y.NYB": "dxy",
        "CL=F": "oil",
        "GC=F": "gold",
    }
    for ticker in BENCHMARK_TICKERS + MACRO_TICKERS:
        df = fetch(ticker, period=period)
        if df.empty:
            continue
        c = df["Close"]
        prefix = names.get(ticker)
        if not prefix:
            continue
        frame = pd.DataFrame(index=df.index)
        frame[f"{prefix}_close"] = c
        if ticker == "^VIX":
            frame["vix_level"] = c
            frame["vix_change_5d"] = c.pct_change(5)
            frame["vix_trend_20"] = c / c.rolling(20).mean() - 1
        elif ticker == "^TNX":
            frame["tnx_level"] = c
            frame["tnx_change_5d"] = c.pct_change(5)
            frame["tnx_trend_50"] = c / c.rolling(50).mean() - 1
        else:
            frame[f"{prefix}_trend_50"] = c / c.rolling(50).mean() - 1
            frame[f"{prefix}_ret_20"] = c.pct_change(20)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    bench = pd.concat(frames, axis=1, sort=False).sort_index()
    neutral = pd.Series(0.5, index=bench.index)

    def above_zero(column):
        return (bench[column] > 0).astype(float) if column in bench else neutral

    def below_zero(column):
        return (bench[column] < 0).astype(float) if column in bench else neutral

    def below_level(column, level):
        return (bench[column] < level).astype(float) if column in bench else neutral

    if {"rsp_close", "spy_close"}.issubset(bench.columns):
        rsp_spy = bench["rsp_close"] / bench["spy_close"]
        bench["rsp_spy_trend_50"] = rsp_spy / rsp_spy.rolling(50).mean() - 1
        bench["rsp_spy_ret_20"] = rsp_spy.pct_change(20)
    if {"iwm_close", "spy_close"}.issubset(bench.columns):
        iwm_spy = bench["iwm_close"] / bench["spy_close"]
        bench["iwm_spy_trend_50"] = iwm_spy / iwm_spy.rolling(50).mean() - 1
        bench["iwm_spy_ret_20"] = iwm_spy.pct_change(20)
    if {"hyg_close", "ief_close"}.issubset(bench.columns):
        hyg_ief = bench["hyg_close"] / bench["ief_close"]
        bench["hyg_ief_trend_50"] = hyg_ief / hyg_ief.rolling(50).mean() - 1
        bench["hyg_ief_ret_20"] = hyg_ief.pct_change(20)

    trend_parts = [
        above_zero("spy_trend_50"),
        above_zero("qqq_trend_50"),
        above_zero("smh_trend_50"),
        above_zero("xli_trend_50"),
        above_zero("xlb_trend_50"),
        above_zero("itb_trend_50"),
    ]
    volatility_parts = [
        below_level("vix_level", 20),
        below_level("vix_level", 25),
        below_zero("vix_change_5d"),
        below_zero("vix_trend_20"),
    ]
    breadth_parts = [
        above_zero("rsp_spy_trend_50"),
        above_zero("rsp_spy_ret_20"),
        above_zero("iwm_spy_trend_50"),
        above_zero("iwm_spy_ret_20"),
    ]
    credit_parts = [
        above_zero("hyg_ief_trend_50"),
        above_zero("hyg_ief_ret_20"),
    ]
    rates_parts = [
        below_level("tnx_trend_50", 0.0),
        below_level("dxy_trend_50", 0.0),
    ]

    bench["trend_score"] = pd.concat(trend_parts, axis=1).mean(axis=1).fillna(neutral)
    bench["volatility_score"] = pd.concat(volatility_parts, axis=1).mean(axis=1).fillna(neutral)
    bench["market_breadth_score"] = pd.concat(breadth_parts, axis=1).mean(axis=1).fillna(neutral)
    bench["credit_stress_score"] = pd.concat(credit_parts, axis=1).mean(axis=1).fillna(neutral)
    bench["rates_sensitivity_score"] = pd.concat(rates_parts, axis=1).mean(axis=1).fillna(neutral)
    bench["risk_on_score"] = (
        bench["trend_score"] * 0.30
        + bench["volatility_score"] * 0.25
        + bench["market_breadth_score"] * 0.25
        + bench["credit_stress_score"] * 0.15
        + bench["rates_sensitivity_score"] * 0.05
    )
    bench = bench.drop(columns=[col for col in bench.columns if col.endswith("_close")], errors="ignore")
    return bench.replace([np.inf, -np.inf], np.nan)


def future_window_extremes(df, horizon_days):
    future_high = df["High"].shift(-1)[::-1].rolling(horizon_days, min_periods=horizon_days).max()[::-1]
    future_low = df["Low"].shift(-1)[::-1].rolling(horizon_days, min_periods=horizon_days).min()[::-1]
    future_close = df["Close"].shift(-horizon_days)
    return future_high / df["Close"] - 1, future_low / df["Close"] - 1, future_close / df["Close"] - 1


def triple_barrier_label(future_max_return, future_min_return, future_return, stop_loss, take_profit):
    """Triple-barrier outcome label (AFML Ch.3), stop-first ordering.

    Mirrors risk_managed_return for a long: a touch of the lower (stop) barrier
    closes the trade at -stop_loss even if the upper barrier was also touched
    within the window. Returns 1 when the risk-managed outcome is positive, else
    0. Unlike fixed-threshold labels, every observation is labeled, so the
    training set includes neutral/no-signal outcomes.
    """
    hit_stop = future_min_return <= -stop_loss
    hit_target = future_max_return >= take_profit
    result = future_return.copy()
    result[hit_target & ~hit_stop] = take_profit
    result[hit_stop] = -stop_loss
    return (result > 0).astype(float)


def market_regime_label(row):
    risk_on = safe_float(row.get("risk_on_score"), 0.5)
    trend = safe_float(row.get("trend_score"), 0.5)
    volatility = safe_float(row.get("volatility_score"), 0.5)
    breadth = safe_float(row.get("market_breadth_score"), 0.5)
    credit = safe_float(row.get("credit_stress_score"), 0.5)
    vix = safe_float(row.get("vix_level"), 20)

    if risk_on <= 0.35 or (trend <= 0.35 and credit <= 0.50) or vix >= 30:
        return "bear"
    if volatility <= 0.35 or vix >= 25:
        return "high_volatility"
    if risk_on >= 0.60 and trend >= 0.55 and breadth >= 0.50 and credit >= 0.50:
        return "trend"
    if 0.40 <= risk_on <= 0.60 and volatility >= 0.50:
        return "mean_reversion"
    return "neutral"


def add_regime_labels(data):
    if data.empty:
        return data
    data["regime_label"] = data.apply(market_regime_label, axis=1)
    return data


def add_peer_features(data, is_live=False):
    """
    Fixed Peer Feature Pipeline.
    Safeguards cross-sectional calculations during live trading by ensuring
    that single-ticker live data requests don't corrupt denominator calculations.
    """
    if data.empty:
        return data

    # If live and we only have one ticker in this slice, peer calculations default
    # to historical baseline distribution rather than division-by-zero NaN.
    grouped = data.groupby("date")

    for col, target in [("ret_5d", "peer_ret_5d"), ("ret_20d", "peer_ret_20d")]:
        peer_sum = grouped[col].transform("sum")
        peer_count = grouped[col].transform("count")

        # Safe structural calculation check
        denom = (peer_count - 1).replace(0, np.nan)
        data[target] = (peer_sum - data[col]) / denom

        # Live fallback if tracking an isolated asset mid-day
        if is_live:
            data[target] = data[target].fillna(0.0)

    data["relative_strength_20d"] = data["ret_20d"] - data["peer_ret_20d"]
    return data


def add_whale_score(data):
    bullish = (
        data["sweep_low_20"]
        + data["failed_breakdown_20"]
        + data["breakout_volume_20"]
        + data["vwap_reclaim"]
        + data["absorption_up"]
        + data["wick_reversal_up"]
        + data["pocket_pivot"]
        + data["range_expansion_up"]
        + data["quiet_accumulation"]
        + ((data["effort_vs_result"] > 4) & (data["close_position"] > 0.60)).astype(float)
        + (data["chaikin_money_flow_20"] > 0.05).astype(float)
        + (data["money_flow_index_14"] > 55).astype(float)
        + (data["accumulation_distribution_10d"] > 0).astype(float)
        + (data["obv_accel"] > 0).astype(float)
        + (data["volume_price_pressure"] > 0).astype(float)
    )
    bearish = (
        data["sweep_high_20"]
        + data["failed_breakout_20"]
        + data["breakdown_volume_20"]
        + data["vwap_rejection"]
        + data["absorption_down"]
        + data["wick_reversal_down"]
        + data["range_expansion_down"]
        + data["quiet_distribution"]
        + ((data["effort_vs_result"] > 4) & (data["close_position"] < 0.40)).astype(float)
        + (data["chaikin_money_flow_20"] < -0.05).astype(float)
        + (data["money_flow_index_14"] < 45).astype(float)
        + (data["accumulation_distribution_10d"] < 0).astype(float)
        + (data["obv_accel"] < 0).astype(float)
        + (data["volume_price_pressure"] < 0).astype(float)
    )
    data["whale_footprint_score"] = (bullish - bearish) / 14
    return data


def setup_candidate_mask(data):
    whale = data.get("whale_footprint_score", 0)
    rs = data.get("relative_strength_20d", 0)
    rsi = data.get("rsi_14", 50)
    atr = data.get("atr_pct", 0)

    bullish_setups = (
        (
            (whale >= 0.20)
            & (rs >= 0.00)
        )
        | (
            (data.get("pocket_pivot", 0) > 0)
            & (data.get("quiet_accumulation", 0) > 0)
        )
        | (
            (data.get("breakout_volume_20", 0) > 0)
            & (data.get("vwap_reclaim", 0) > 0)
        )
        | (
            (data.get("sweep_low_20", 0) > 0)
            & (data.get("wick_reversal_up", 0) > 0)
        )
        | (
            (rsi < 35)
            & (whale > 0)
            & (rs > -0.02)
        )
    )

    bearish_setups = (
        (
            (whale <= -0.20)
            & (rs <= 0.00)
        )
        | (
            (data.get("breakdown_volume_20", 0) > 0)
            & (data.get("vwap_rejection", 0) > 0)
        )
        | (
            (data.get("sweep_high_20", 0) > 0)
            & (data.get("wick_reversal_down", 0) > 0)
        )
        | (
            (data.get("quiet_distribution", 0) > 0)
            & (data.get("volume_dry_up", 0) > 0)
        )
        | (
            (rsi > 70)
            & (whale < 0)
            & (rs < 0.02)
        )
    )

    tradable_volatility = (atr > 0.015) & (atr < 0.12)

    return (bullish_setups | bearish_setups) & tradable_volatility


def make_dataset(tickers, period, horizon_days, return_threshold,
                 training_mode=DEFAULT_TRAINING_MODE,
                 labeling="fixed", stop_loss=DEFAULT_STOP_LOSS, take_profit=DEFAULT_TAKE_PROFIT,
                 is_live=False):
    """
    Upgraded Data Pipeline Engine.
    Prevents uncompleted daily bar ingestion and aligns the inference data
    distribution with the walk-forward model constraints.
    """
    frames = []
    section("FETCHING MARKET REGIME DATA")
    benchmark_features = make_benchmark_features(period)

    for ticker in tickers:
        print(f" Fetching {ticker:<10} {period}")
        df = fetch(ticker, period=period)

        if len(df) < 260:
            print(f" {Fore.YELLOW}Skipping {ticker}: need more history.{Style.RESET_ALL}")
            continue

        # --- CRITICAL FIX 1: THE INCOMPLETE LIVE BAR TRAP ---
        # If trading live on daily bars, the last bar is actively moving (incomplete).
        # We drop the live candle or shift back to execute safely on finalized metrics.
        if is_live:
            current_time = dt.datetime.now()
            # If market is still open, drop the incomplete day's row
            if current_time.hour < 16:
                df = df.iloc[:-1].copy()

        feats = make_features(df)
        if not benchmark_features.empty:
            feats = feats.join(benchmark_features, how="left")

        fundamental = fetch_fundamental_snapshot(ticker)
        for col, value in fundamental.items():
            feats[col] = value

        kronos_feats = kronos_forecast_features(ticker, df)
        for col, value in kronos_feats.items():
            feats[col] = value

        future_max_return, future_min_return, future_return = future_window_extremes(df, horizon_days)

        if labeling == "triple_barrier":
            label = triple_barrier_label(future_max_return, future_min_return, future_return, stop_loss, take_profit)
        else:
            label = pd.Series(np.nan, index=df.index)
            label[future_return > return_threshold] = 1
            label[future_return < -return_threshold] = 0

        data = feats.copy()
        data["ticker"] = ticker
        data["date"] = data.index
        data["future_return"] = future_return
        data["future_max_return"] = future_max_return
        data["future_min_return"] = future_min_return
        data["target"] = label
        frames.append(data)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames).sort_values("date").reset_index(drop=True)

    # Process peer and whale metrics with live-aware protection matrices
    data = add_peer_features(data, is_live=is_live)
    data = add_whale_score(data)
    data = add_regime_labels(data)
    data["setup_candidate"] = setup_candidate_mask(data).astype(int)

    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0

    data[FEATURE_COLUMNS] = (
        data.groupby("ticker")[FEATURE_COLUMNS]
        .ffill()
        .fillna(0)
    )

    # Live execution shouldn't drop rows because target variable labels are missing out-of-sample
    if not is_live:
        data = data.dropna(
            subset=FEATURE_COLUMNS + ["target", "future_return", "future_max_return", "future_min_return"])
    else:
        data = data.dropna(subset=FEATURE_COLUMNS)

    # --- CRITICAL FIX 2: THE "SETUPS ONLY" DISTRIBUTION SHIFT ---
    # Enforces that live inference ONLY considers setups identical to training parameters
    if training_mode == "setups" or is_live:
        before = len(data)
        data = data[data["setup_candidate"] == 1].copy()
        kept = len(data) / before * 100 if before else 0
        if not is_live:
            print(f" Setup-candidate filter: {len(data):,}/{before:,} rows kept ({kept:.1f}%)")

    if not data.empty and "target" in data.columns:
        # Fill training target labels seamlessly
        data["target"] = data["target"].fillna(0).astype(int)

    return data

def run_live_inference_cycle(tickers, model_pipeline):
    """
    Live Execution Coordinator Entry Point.
    Enforces backtest parity live by processing the universe simultaneously.
    """
    import datetime as dt
    print(f"\n[{dt.datetime.now()}] Launching Safe Live Inference Pipeline...")

    # Process full asset vector simultaneously to satisfy peer matrix dependencies
    live_dataset = make_dataset(
        tickers=tickers,
        period="1y",
        horizon_days=DEFAULT_HORIZON_DAYS,
        return_threshold=DEFAULT_RETURN_THRESHOLD,
        training_mode="setups",
        is_live=True
    )

    if live_dataset.empty:
        print("Inference complete: No live assets passed candidate setup parameters today.")
        return []

    # Filter for the most recent completed trading bar snapshot per ticker
    latest_signals = live_dataset.groupby("ticker").last().reset_index()
    X_live = latest_signals[ACTIVE_FEATURE_COLUMNS]

    # Generate classification probability distributions matching walk-forward environments
    probabilities = model_pipeline.predict_proba(X_live)[:, 1]
    live_records = []

    for idx, row in latest_signals.iterrows():
        prob_up = float(probabilities[idx])
        prob_up = regime_probability_overlay(prob_up, row)
        signal = signal_from_probability(prob_up, DEFAULT_PROB_UP, DEFAULT_PROB_DOWN)

        if signal != 0:
            if DEFAULT_TRAINING_MODE == "setups":
                signal = strict_setup_signal({**row.to_dict(), "signal": signal})

            if signal != 0:
                pos_size = position_size(prob_up, signal, DEFAULT_PROB_UP, DEFAULT_PROB_DOWN)

                record = {
                    "ticker": row["ticker"],
                    "price": float(row["Close"]),
                    "signal": "LONG" if signal == 1 else "SHORT",
                    "prob_up": prob_up,
                    "position_size": pos_size,
                    "rsi": float(row.get("rsi_14", 50)),
                    "whale_score": float(row.get("whale_footprint_score", 0)),
                    "risk_on_score": float(row.get("risk_on_score", 0.5)),
                    "fundamental_score": float(row.get("fund_rec_score", 0)),
                    "horizon_days": DEFAULT_HORIZON_DAYS,
                }
                live_records.append(record)

    if live_records:
        print(f"Success: {len(live_records)} execution vectors matched validation properties.")
        log_live_predictions_to_db(live_records)
    else:
        print("Evaluation complete: Active assets tracked, but edge margins did not justify exposure.")

    return live_records

class AverageProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators=None):
        self.estimators = estimators

    def fit(self, X, y):
        base_estimators = self.estimators or [
            (
                "logistic",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                    ]
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=12,
                    max_depth=7,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.035,
                    max_leaf_nodes=15,
                    l2_regularization=0.25,
                    random_state=42,
                ),
            ),
        ]
        self.models_ = []
        for name, estimator in base_estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.models_.append((name, fitted))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for _, model in self.models_]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AttentionNetClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible attention-based neural classifier.

    A learnable soft-attention layer over the feature dimension feeds a small
    MLP. The fitted attention weights give per-feature attribution for
    interpretability. Torch is imported lazily so the module still loads when
    torch is unavailable (the candidate is simply skipped).
    """

    def __init__(self, hidden_dim=32, dropout=0.3, learning_rate=1e-3, epochs=50, batch_size=256, random_state=42):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        import torch

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        y_arr = y_arr.astype(np.float64)

        torch.manual_seed(self.random_state)
        self.net_ = _AttentionNet(Xs.shape[1], self.hidden_dim, self.dropout)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y_arr, dtype=torch.float32)
        n_samples = Xt.shape[0]
        for _ in range(self.epochs):
            self.net_.train()
            permutation = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                indices = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                logits = self.net_(Xt[indices])
                loss = loss_fn(logits, yt[indices])
                loss.backward()
                optimizer.step()

        self.net_.eval()
        with torch.no_grad():
            self.attention_mean_ = self.net_.attention(Xt).mean(dim=0).numpy()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        self.net_.eval()
        Xs = self.scaler_.transform(X)
        with torch.no_grad():
            logits = self.net_(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()
        try:
            from scipy.special import expit
        except Exception:
            logits = np.clip(logits, -50, 50)
            probs = 1 / (1 + np.exp(-logits))
        else:
            probs = expit(logits)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def attention_weights(self, X=None):
        if not hasattr(self, "net_"):
            return None
        if X is None:
            return getattr(self, "attention_mean_", None)
        import torch

        self.net_.eval()
        Xs = self.scaler_.transform(X)
        with torch.no_grad():
            alpha = self.net_.attention(torch.tensor(Xs, dtype=torch.float32)).mean(dim=0).numpy()
        return alpha


if torch is not None:

    class _AttentionNet(torch.nn.Module):
        """Module-level attention net so trained instances can be pickled/saved."""

        def __init__(self, n_features, hidden_dim, dropout):
            super().__init__()
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(n_features, n_features),
                torch.nn.Softmax(dim=1),
            )
            self.model = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            alpha = self.attention(x)
            return self.model(x * alpha).squeeze(-1)

else:
    _AttentionNet = None


def candidate_models():
    models = {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=12,
            max_depth=7,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.25,
            random_state=42,
        ),
        "ensemble": AverageProbabilityEnsemble(),
    }
    try:
        import torch  # noqa: F401

        models["attention_net"] = AttentionNetClassifier()
    except Exception:
        pass
    return models


def signal_from_probability(prob_up, up_threshold, down_threshold):
    if prob_up >= up_threshold:
        return 1
    if prob_up <= down_threshold:
        return -1
    return 0


def apply_thresholds(rows, up_threshold, down_threshold):
    thresholded = []
    for row in rows:
        updated = row.copy()
        updated["signal"] = signal_from_probability(row["prob_up"], up_threshold, down_threshold)
        thresholded.append(updated)
    return thresholded


def regime_probability_overlay(raw_prob, row):
    def clean(value, default):
        try:
            return default if pd.isna(value) else float(value)
        except Exception:
            return default

    risk_on = clean(row.get("risk_on_score", 0.5), 0.5)
    breadth = clean(row.get("market_breadth_score", 0.5), 0.5)
    credit = clean(row.get("credit_stress_score", 0.5), 0.5)
    volatility = clean(row.get("volatility_score", 0.5), 0.5)
    whale = clean(row.get("whale_footprint_score", row.get("whale_score", 0)), 0)

    adj = 0.0
    adj += (risk_on - 0.5) * 0.04
    adj += (breadth - 0.5) * 0.03
    adj += (credit - 0.5) * 0.03
    adj += whale * 0.06

    if breadth < 0.25:
        adj -= 0.03
    if credit < 0.25:
        adj -= 0.02
    if volatility < 0.25:
        adj -= 0.03

    return float(np.clip(raw_prob + adj, 0.05, 0.95))


def position_size(prob_up, signal, up_threshold, down_threshold, max_position_size=DEFAULT_MAX_POSITION_SIZE):
    if signal == 1:
        edge = max(0, prob_up - up_threshold)
    elif signal == -1:
        edge = max(0, down_threshold - prob_up)
    else:
        return 0
    if edge >= 0.12:
        size = 1.0
    elif edge >= 0.06:
        size = 0.75
    else:
        size = 0.35
    return min(size, max_position_size)


def strict_setup_signal(row):
    signal = row["signal"]
    if signal == 0:
        return 0

    whale_score = row.get("whale_footprint_score", 0)
    risk_on_score = row.get("risk_on_score", 0.5)
    trend_score = row.get("trend_score", 0.5)
    volatility_score = row.get("volatility_score", 0.5)
    breadth_score = row.get("market_breadth_score", 0.5)
    credit_score = row.get("credit_stress_score", 0.5)
    relative_strength = row.get("relative_strength_20d", 0)
    if signal == 1:
        aligned = (
            whale_score > 0
            and risk_on_score >= 0.55
            and trend_score >= 0.50
            and breadth_score >= 0.50
            and credit_score >= 0.50
            and volatility_score >= 0.50
            and relative_strength >= -0.02
        )
    else:
        aligned = (
            whale_score < 0
            and risk_on_score <= 0.45
            and (trend_score <= 0.50 or breadth_score <= 0.50)
            and (credit_score <= 0.50 or volatility_score <= 0.50)
            and relative_strength <= 0.02
        )
    return signal if aligned else 0


def sector_for_ticker(ticker):
    return SECTOR_MAP.get(str(ticker).upper(), "other")


def resolve_risk_levels(row, stop_loss, take_profit, risk_mode="fixed"):
    if risk_mode != "atr":
        return stop_loss, take_profit
    atr_pct = row.get("atr_pct", np.nan)
    if pd.isna(atr_pct) or atr_pct <= 0:
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    actual_stop = min(MAX_ATR_STOP, max(MIN_ATR_STOP, atr_pct * stop_loss))
    reward_ratio = take_profit / stop_loss if stop_loss else 2.0
    return actual_stop, actual_stop * reward_ratio


def risk_managed_return(row, stop_loss, take_profit, execution_mode="stop_first", stress_slippage_bps=0, risk_mode="fixed"):
    signal = row["signal"]
    if signal == 0:
        return 0

    stop_loss, take_profit = resolve_risk_levels(row, stop_loss, take_profit, risk_mode)
    final_return = row["future_return"]
    max_return = row.get("future_max_return", final_return)
    min_return = row.get("future_min_return", final_return)

    slippage = stress_slippage_bps / 10000

    if signal == 1:
        hit_stop = min_return <= -stop_loss
        hit_target = max_return >= take_profit
        if hit_stop and hit_target:
            result = take_profit if execution_mode == "target_first" else -stop_loss
        elif hit_stop:
            result = -stop_loss
        elif hit_target:
            result = take_profit
        else:
            result = final_return
        return result - (slippage * 2 if hit_stop else slippage)

    hit_stop = max_return >= stop_loss
    hit_target = min_return <= -take_profit
    if hit_stop and hit_target:
        result = take_profit if execution_mode == "target_first" else -stop_loss
    elif hit_stop:
        result = -stop_loss
    elif hit_target:
        result = take_profit
    else:
        result = -final_return
    return result - (slippage * 2 if hit_stop else slippage)


def portfolio_return_series(
    traded,
    fee_bps,
    max_portfolio_exposure=DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
    daily_stop=DEFAULT_DAILY_STOP,
    equity_breaker=DEFAULT_EQUITY_BREAKER,
    max_consecutive_loss_days=DEFAULT_MAX_CONSECUTIVE_LOSS_DAYS,
    loss_cooldown_days=DEFAULT_LOSS_COOLDOWN_DAYS,
):
    if traded.empty:
        return pd.Series(dtype=float)

    working = traded.copy()
    working["date"] = pd.to_datetime(working["date"])
    working["sector"] = working["ticker"].apply(sector_for_ticker)
    working["scaled_size"] = working["position_size"].astype(float)
    working["raw_trade_return"] = working["raw_trade_return"].astype(float)

    # Portfolio-level exposure cap (per day). Vectorized: compute each day's
    # requested exposure, scale down only the days that exceed the cap.
    requested = working["scaled_size"].abs().groupby(working["date"]).sum()
    portfolio_scale = (max_portfolio_exposure / requested).clip(upper=1.0)
    working["scaled_size"] = working["scaled_size"] * working["date"].map(portfolio_scale)

    # Sector-level exposure cap (per day per sector), applied on top.
    sector_exposure = working["scaled_size"].abs().groupby([working["date"], working["sector"]]).sum()
    sector_scale = (max_sector_exposure / sector_exposure).clip(upper=1.0).rename("sector_scale")
    working = working.join(sector_scale, on=["date", "sector"])
    working["scaled_size"] = working["scaled_size"] * working["sector_scale"]

    # Per-day net return after position sizing and fees.
    day_return = (working["raw_trade_return"] * working["scaled_size"]).groupby(working["date"]).sum()
    day_return = day_return - (fee_bps / 10000) * working["scaled_size"].abs().groupby(working["date"]).sum()
    if daily_stop:
        day_return = day_return.where(day_return >= -daily_stop, -daily_stop)

    returns = []
    equity = 1.0
    peak = 1.0
    breaker_tripped = False
    breaker_trigger_date = None
    days_after_breaker = 0
    consecutive_losses = 0
    cooldown_remaining = 0
    cooldown_trigger_count = 0
    cooldown_days_used = 0
    for date, ret in day_return.items():
        if breaker_tripped:
            returns.append(0.0)
            days_after_breaker += 1
            continue
        if cooldown_remaining > 0:
            returns.append(0.0)
            cooldown_remaining -= 1
            cooldown_days_used += 1
            continue

        returns.append(float(ret))
        if ret < 0:
            consecutive_losses += 1
        elif ret > 0:
            consecutive_losses = 0
        if (
            max_consecutive_loss_days
            and loss_cooldown_days
            and consecutive_losses >= max_consecutive_loss_days
        ):
            cooldown_remaining = loss_cooldown_days
            cooldown_trigger_count += 1
            consecutive_losses = 0

        equity *= 1 + ret
        peak = max(peak, equity)
        if equity_breaker and equity / peak - 1 <= -equity_breaker:
            breaker_tripped = True
            breaker_trigger_date = date

    series = pd.Series(
        returns,
        index=day_return.index,
    ).sort_index()
    series.attrs["breaker_triggered"] = breaker_tripped
    series.attrs["breaker_trigger_date"] = breaker_trigger_date
    series.attrs["days_after_breaker"] = days_after_breaker
    series.attrs["cooldown_trigger_count"] = cooldown_trigger_count
    series.attrs["cooldown_days_used"] = cooldown_days_used
    return series


def add_trade_rank(traded):
    ranked = traded.copy()
    whale = ranked.get("whale_footprint_score", pd.Series(0, index=ranked.index))
    rs = ranked.get("relative_strength_20d", pd.Series(0, index=ranked.index))
    ranked["prob_edge"] = np.where(
        ranked["signal"] == 1,
        ranked["prob_up"] - 0.5,
        0.5 - ranked["prob_up"],
    )
    ranked["whale_alignment"] = np.where(ranked["signal"] == 1, whale, -whale)
    ranked["rs_alignment"] = np.where(ranked["signal"] == 1, rs, -rs)
    ranked["setup_score"] = (
        ranked.get("pocket_pivot", pd.Series(0, index=ranked.index)) * 15
        + ranked.get("breakout_volume_20", pd.Series(0, index=ranked.index)) * 10
        + ranked.get("quiet_accumulation", pd.Series(0, index=ranked.index)) * 8
        + ranked.get("vwap_reclaim", pd.Series(0, index=ranked.index)) * 8
        + ranked.get("setup_candidate", pd.Series(0, index=ranked.index)) * 5
    )
    ranked["trade_rank"] = (
        ranked["prob_edge"].clip(lower=0) * 100
        + ranked["whale_alignment"].clip(lower=0) * 30
        + ranked["rs_alignment"].clip(lower=0) * 40
        + ranked["position_size"].abs() * 10
        + ranked["setup_score"]
    )
    return ranked


def limit_daily_trades(
    traded,
    max_daily_trades=DEFAULT_MAX_DAILY_TRADES,
    max_daily_sector_trades=DEFAULT_MAX_DAILY_SECTOR_TRADES,
):
    if traded.empty or not max_daily_trades:
        return traded

    working = add_trade_rank(traded)
    working["date"] = pd.to_datetime(working["date"])
    working["sector"] = working["ticker"].apply(sector_for_ticker)
    if "trade_heat" in working.columns:
        working["_heat"] = working["trade_heat"].abs()
    else:
        working["_heat"] = (working["position_size"].abs() * DEFAULT_STOP_LOSS)

    # Per-day trade cap: reduced to 1 in risk-off sessions.
    if "risk_on_score" in working.columns:
        risk_off = working.groupby("date")["risk_on_score"].transform("mean") < 0.45
        working["_day_cap"] = np.where(risk_off, min(max_daily_trades, 1), max_daily_trades)
    else:
        working["_day_cap"] = max_daily_trades

    # Per-(day, sector) cap: keep only the top max_daily_sector_trades by rank.
    working["_rank"] = working.groupby(["date", "sector"])["trade_rank"].rank(
        ascending=False, method="first"
    )
    working = working[working["_rank"] <= max_daily_sector_trades]
    working = working.sort_values(["date", "trade_rank"], ascending=[True, False])

    # Per-day greedy: walk candidates in rank order, respecting the portfolio
    # heat budget and the daily trade cap.
    kept_idx = []
    for _, group in working.groupby("date", sort=False):
        heats = group["_heat"].to_numpy()
        cap = int(group["_day_cap"].iloc[0])
        heat_used = 0.0
        count = 0
        for idx, heat in zip(group.index, heats):
            if heat_used + heat > DEFAULT_MAX_PORTFOLIO_HEAT and count > 0:
                continue
            kept_idx.append(idx)
            heat_used += heat
            count += 1
            if count >= cap:
                break

    if not kept_idx:
        return working.iloc[0:0].copy()
    kept = working.loc[kept_idx].drop(columns=["_heat", "_day_cap", "_rank"])
    return kept.sort_values(["date", "trade_rank"], ascending=[True, False])



def profit_factor_from_returns(returns):
    values = pd.Series(returns).dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses <= 0:
        return np.inf if gains > 0 else 0
    return gains / losses


def max_drawdown_from_returns(returns):
    values = pd.Series(returns).fillna(0)
    if values.empty:
        return 0
    equity = (1 + values).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def annualized_sharpe(returns, periods_per_year=252):
    returns = pd.Series(returns).dropna().astype(float)
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _sr_estimator_stats(returns):
    """Per-period Sharpe estimator plus skew/excess-kurtosis (ML4AM Ch.6)."""
    returns = pd.Series(returns).dropna().astype(float)
    n = len(returns)
    if n < 3:
        return None
    sd = returns.std(ddof=1)
    if sd <= 0:
        return None
    return {
        "n": n,
        "sr": float(returns.mean() / sd),
        "skew": float(returns.skew()),
        "kurt": float(returns.kurt()),
    }


def probabilistic_sharpe_ratio(returns, sr_threshold=0.0):
    """Probability the true Sharpe exceeds sr_threshold (Bailey-Lopez de Prado)."""
    stats = _sr_estimator_stats(returns)
    if stats is None:
        return 0.0
    denom = math.sqrt(max(1 - stats["skew"] * stats["sr"] + (stats["kurt"] - 1) / 4 * stats["sr"] ** 2, 1e-12))
    z = (stats["sr"] - sr_threshold) * math.sqrt(stats["n"] - 1) / denom
    return float(_norm_cdf(z))


def deflated_sharpe_ratio(returns, trials=DEFAULT_SELECTION_TRIALS):
    """Deflated Sharpe Ratio corrected for multiple trials (ML4AM Ch.6)."""
    stats = _sr_estimator_stats(returns)
    if stats is None or trials < 1:
        return 0.0
    try:
        from scipy.stats import norm
    except Exception:
        return 0.0
    euler = 0.5772156649015328606
    variance = max((1 - stats["skew"] * stats["sr"] + (stats["kurt"] - 1) / 4 * stats["sr"] ** 2) / (stats["n"] - 1), 1e-12)
    sd_sr = math.sqrt(variance)
    emax = sd_sr * ((1 - euler) * norm.ppf(1 - 1.0 / trials) + euler * norm.ppf(1 - 1.0 / (trials * math.e)))
    denom = math.sqrt(max(1 - stats["skew"] * stats["sr"] + (stats["kurt"] - 1) / 4 * stats["sr"] ** 2, 1e-12))
    z = (stats["sr"] - emax) * math.sqrt(stats["n"] - 1) / denom
    return float(norm.cdf(z))


def portfolio_risk_metrics(returns, trials=DEFAULT_SELECTION_TRIALS):
    """(per-period sharpe, annualized sharpe, PSR, DSR) for a return series."""
    returns = pd.Series(returns).dropna().astype(float)
    if len(returns) < 2:
        return 0.0, 0.0, 0.0, 0.0
    sharpe = float(returns.mean() / (returns.std(ddof=1) + 1e-12))
    annual_sr = annualized_sharpe(returns)
    psr = probabilistic_sharpe_ratio(returns, sr_threshold=0.0)
    dsr = deflated_sharpe_ratio(returns, trials=trials)
    return sharpe, annual_sr, psr, dsr


def dynamic_position_size_from_row(row, base_size, target_trade_risk=DEFAULT_TARGET_TRADE_RISK):
    atr = safe_float(row.get("atr_pct"), 0)
    if atr <= 0:
        vol_size = base_size
    else:
        vol_size = min(base_size, max(DEFAULT_MIN_POSITION_SIZE, target_trade_risk / atr))

    risk_on = safe_float(row.get("risk_on_score"), 0.5)
    signal = int(row.get("signal", 0))
    regime_scale = 1.0
    # Symmetric regime scaling: a long in a weak (risk-off) regime is scaled down
    # by the same factors as a short in a strong (risk-on) regime, and vice versa,
    # so neither direction gets a structurally easier ride through the regime filter.
    if signal == 1:
        if risk_on < 0.40:
            regime_scale = 0.45
        elif risk_on < 0.50:
            regime_scale = 0.70
        elif risk_on > 0.65:
            regime_scale = 1.15
    elif signal == -1:
        if risk_on > 0.65:
            regime_scale = 0.45
        elif risk_on > 0.55:
            regime_scale = 0.70
        elif risk_on < 0.40:
            regime_scale = 1.15
    return float(min(base_size, max(0, vol_size * regime_scale)))


def add_institutional_position_controls(traded, max_position_size):
    if traded.empty:
        return traded
    adjusted = traded.copy()
    adjusted["base_position_size"] = adjusted["position_size"]
    adjusted["position_size"] = adjusted.apply(
        lambda row: dynamic_position_size_from_row(row, row["base_position_size"] or max_position_size),
        axis=1,
    )
    adjusted["trade_heat"] = adjusted["position_size"].abs() * adjusted.get("stop_loss_used", DEFAULT_STOP_LOSS)
    return adjusted


def calibration_bins_report(rows_or_df, min_bin_size=10):
    if rows_or_df is None:
        return []
    df = pd.DataFrame(rows_or_df).copy()
    if df.empty or "prob_up" not in df or "target" not in df:
        return []
    bins = np.arange(0, 1.01, 0.10)
    report = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high >= 1:
            mask = (df["prob_up"] >= low) & (df["prob_up"] <= high)
        else:
            mask = (df["prob_up"] >= low) & (df["prob_up"] < high)
        count = int(mask.sum())
        if count < min_bin_size:
            continue
        predicted = float(df.loc[mask, "prob_up"].mean())
        actual = float(df.loc[mask, "target"].mean())
        report.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "count": count,
            "predicted": predicted,
            "actual": actual,
            "error": actual - predicted,
        })
    return report


def monte_carlo_trade_shuffle(traded, runs=1000, random_state=42):
    """Monte Carlo audit with two separate checks.

    1) Sequencing shuffle: same trades, different order. Final equity is order-invariant,
       so this is mainly useful for drawdown distribution.
    2) Bootstrap resample: samples trades with replacement. This gives a real equity
       distribution and exposes dependency on a few lucky trades.
    """
    if traded is None or traded.empty:
        return {}

    return_col = "strategy_return" if "strategy_return" in traded else "raw_trade_return"
    if return_col not in traded:
        return {}

    returns = pd.Series(traded[return_col]).dropna().astype(float).to_numpy()
    if len(returns) < 10:
        return {"runs": 0, "note": "too few trades"}

    rng = np.random.default_rng(random_state)
    shuffle_dds = []
    bootstrap_equities = []
    bootstrap_dds = []

    for _ in range(runs):
        # Same trades, randomized order: final equity stays the same, DD changes.
        shuffled = rng.permutation(returns)
        equity_curve = np.cumprod(1 + shuffled)
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / peaks - 1
        shuffle_dds.append(float(drawdowns.min()))

        # Resampled trades: both final equity and DD become distributions.
        sampled = rng.choice(returns, size=len(returns), replace=True)
        sample_equity_curve = np.cumprod(1 + sampled)
        sample_peaks = np.maximum.accumulate(sample_equity_curve)
        sample_drawdowns = sample_equity_curve / sample_peaks - 1
        bootstrap_equities.append(float(sample_equity_curve[-1]))
        bootstrap_dds.append(float(sample_drawdowns.min()))

    actual_equity = float(np.prod(1 + returns))
    return {
        "runs": int(runs),
        "trades": int(len(returns)),
        "actual_equity": actual_equity,
        "bootstrap_median_equity": float(np.median(bootstrap_equities)),
        "bootstrap_p05_equity": float(np.percentile(bootstrap_equities, 5)),
        "bootstrap_p95_equity": float(np.percentile(bootstrap_equities, 95)),
        "shuffle_median_dd": float(np.median(shuffle_dds)),
        "shuffle_p05_dd": float(np.percentile(shuffle_dds, 5)),
        "shuffle_worst_dd": float(np.min(shuffle_dds)),
        "bootstrap_median_dd": float(np.median(bootstrap_dds)),
        "bootstrap_p05_dd": float(np.percentile(bootstrap_dds, 5)),
        "bootstrap_worst_dd": float(np.min(bootstrap_dds)),
        "ruin_prob_20pct": float(np.mean(np.array(bootstrap_dds) <= -0.20)),
    }

def feature_drift_audit(train_df, final_df, feature_cols, z_limit=DEFAULT_FEATURE_DRIFT_Z_LIMIT):
    if train_df is None or final_df is None or train_df.empty or final_df.empty:
        return []
    rows = []
    for col in feature_cols:
        if col not in train_df or col not in final_df:
            continue
        train_mean = float(train_df[col].mean())
        train_std = float(train_df[col].std())
        final_mean = float(final_df[col].mean())
        if not np.isfinite(train_std) or train_std <= 1e-9:
            continue
        z = abs(final_mean - train_mean) / train_std
        if z >= z_limit:
            rows.append({"feature": col, "train_mean": train_mean, "final_mean": final_mean, "z_drift": float(z)})
    return sorted(rows, key=lambda item: item["z_drift"], reverse=True)


def leakage_correlation_audit(data, feature_cols, threshold=0.35):
    if data is None or data.empty or "future_return" not in data:
        return []
    rows = []
    for col in feature_cols:
        if col not in data:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = data[col].corr(data["future_return"])
        if pd.notna(corr) and abs(corr) >= threshold:
            rows.append({"feature": col, "corr_to_future_return": float(corr)})
    return sorted(rows, key=lambda item: abs(item["corr_to_future_return"]), reverse=True)


def feature_redundancy_audit(data, feature_cols, threshold=0.92):
    if data is None or data.empty:
        return []
    available = [col for col in feature_cols if col in data]
    if len(available) < 2:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = data[available].corr().abs()
    rows = []
    for i, left in enumerate(available):
        for right in available[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "corr": float(value)})
    return sorted(rows, key=lambda item: item["corr"], reverse=True)


def temporal_stability_audit(traded):
    if traded is None or traded.empty:
        return []
    df = traded.copy()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for year, group in df.groupby(df["date"].dt.year):
        rows.append({
            "year": int(year),
            "trades": int(len(group)),
            "accuracy": float(group["correct"].mean() * 100) if "correct" in group else 0,
            "avg_return": float(group["strategy_return"].mean() * 100),
            "profit_factor": profit_factor_from_returns(group["strategy_return"]),
            "max_drawdown": max_drawdown_from_returns(group["strategy_return"]) * 100,
        })
    return rows


def ticker_dependency_audit(traded, concentration_limit=0.30):
    if traded is None or traded.empty:
        return []
    total_profit = traded["strategy_return"].sum()
    if abs(total_profit) <= 1e-12:
        return []
    stats = traded.groupby("ticker")["strategy_return"].agg(["count", "mean", "sum"]).reset_index()
    stats["contribution"] = stats["sum"] / total_profit
    flagged = stats[stats["contribution"].abs() >= concentration_limit].copy()
    return flagged.sort_values("contribution", key=lambda s: s.abs(), ascending=False).to_dict("records")


def print_phase_audit_report(title, rows, headers=None, max_rows=10):
    if not rows:
        return
    section(title)
    print(tabulate(rows[:max_rows], headers=headers or "keys", tablefmt="simple"))

def evaluate_predictions(
    rows,
    fee_bps,
    up_threshold=None,
    down_threshold=None,
    stop_loss=None,
    take_profit=None,
    max_position_size=DEFAULT_MAX_POSITION_SIZE,
    max_portfolio_exposure=DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
    daily_stop=DEFAULT_DAILY_STOP,
    equity_breaker=DEFAULT_EQUITY_BREAKER,
    strict_filter=False,
    execution_mode="stop_first",
    stress_slippage_bps=0,
    risk_mode="fixed",
    max_daily_trades=DEFAULT_MAX_DAILY_TRADES,
    max_daily_sector_trades=DEFAULT_MAX_DAILY_SECTOR_TRADES,
    max_consecutive_loss_days=DEFAULT_MAX_CONSECUTIVE_LOSS_DAYS,
    loss_cooldown_days=DEFAULT_LOSS_COOLDOWN_DAYS,
    apply_quality_gate=True,
    meta_model=None,
    meta_threshold=None,
    fast=False,
    trials=DEFAULT_SELECTION_TRIALS,
):
    if not rows:
        return {}
    if up_threshold is not None and down_threshold is not None:
        rows = apply_thresholds(rows, up_threshold, down_threshold)
    out = pd.DataFrame(rows)

    # Backtest/live parity fix: live trading only actually takes a trade when
    # actionable_signal() clears the same A/B quality gate used for live signals
    # (see live_prediction_records). Previously this backtest traded every row
    # where the raw threshold fired, which is a materially larger and easier
    # set of trades than what a user would ever see live. Applying the same
    # gate here means reported accuracy/profit-factor/drawdown reflect the
    # trades a live user would actually be shown.
    if apply_quality_gate and not out.empty:
        def _gated_signal(row):
            whale_score = float(row.get("whale_footprint_score", 0))
            risk_on_score = float(row.get("risk_on_score", 0.5))
            quality = trade_quality(row["prob_up"], row["signal"], whale_score, risk_on_score, row)
            setup_candidate = bool(row.get("setup_candidate", 1))
            return actionable_signal(row["signal"], quality, setup_candidate, row)

        out["quality"] = out.apply(
            lambda row: trade_quality(
                row["prob_up"],
                row["signal"],
                float(row.get("whale_footprint_score", 0)),
                float(row.get("risk_on_score", 0.5)),
                row,
            ),
            axis=1,
        )
        out["signal"] = out.apply(_gated_signal, axis=1)

    if strict_filter and not out.empty:
        out["signal"] = out.apply(strict_setup_signal, axis=1)
    traded = out[out["signal"] != 0].copy()
    fee = fee_bps / 10000

    if traded.empty:
        return {
            "rows": out,
            "traded": traded,
            "portfolio_returns": pd.Series(dtype=float),
            "accuracy": 0,
            "coverage": 0,
            "avg_return": 0,
            "profit_factor": 0,
            "portfolio_profit_factor": 0,
            "max_drawdown": 0,
            "brier": brier_score_loss(out["target"], out["prob_up"]),
            "sharpe": 0,
            "annualized_sharpe": 0,
            "psr": 0,
            "dsr": 0,
            "up_threshold": up_threshold,
            "down_threshold": down_threshold,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_position_size": max_position_size,
            "max_portfolio_exposure": max_portfolio_exposure,
            "max_sector_exposure": max_sector_exposure,
            "daily_stop": daily_stop,
            "equity_breaker": equity_breaker,
            "strict_filter": strict_filter,
            "execution_mode": execution_mode,
            "stress_slippage_bps": stress_slippage_bps,
            "risk_mode": risk_mode,
            "max_daily_trades": max_daily_trades,
            "max_daily_sector_trades": max_daily_sector_trades,
            "max_consecutive_loss_days": max_consecutive_loss_days,
            "loss_cooldown_days": loss_cooldown_days,
            "breaker_triggered": False,
            "breaker_trigger_date": None,
            "days_after_breaker": 0,
            "cooldown_trigger_count": 0,
            "cooldown_days_used": 0,
        }

    stop_loss = stop_loss or DEFAULT_STOP_LOSS
    take_profit = take_profit or DEFAULT_TAKE_PROFIT
    traded["position_size"] = traded.apply(
        lambda row: position_size(
            row["prob_up"],
            row["signal"],
            up_threshold or 0.60,
            down_threshold or 0.40,
            max_position_size,
        ),
        axis=1,
    )
    traded["stop_loss_used"] = traded.apply(
        lambda row: resolve_risk_levels(row, stop_loss, take_profit, risk_mode)[0],
        axis=1,
    )
    traded = add_institutional_position_controls(traded, max_position_size)
    traded["raw_trade_return"] = traded.apply(
        lambda row: risk_managed_return(row, stop_loss, take_profit, execution_mode, stress_slippage_bps, risk_mode),
        axis=1,
    )
    traded["strategy_return"] = (
        traded["position_size"] * traded["raw_trade_return"]
        - fee * traded["position_size"].abs()
    )
    traded["correct"] = np.where(
        traded["signal"] == 1,
        traded["future_return"] > 0,
        traded["future_return"] < 0,
    )
    if "prob_sigma" in traded.columns:
        traded["position_size"] = traded["position_size"] * traded["prob_sigma"].apply(
            lambda s: float(np.clip(1 - 2.5 * s, 0, 1))
        )
    if meta_model is not None:
        traded = apply_meta_gate_and_size(traded, meta_model, meta_threshold)
        traded["strategy_return"] = (
            traded["position_size"] * traded["raw_trade_return"]
            - fee * traded["position_size"].abs()
        )
    signal_count = len(traded)
    traded = limit_daily_trades(traded, max_daily_trades, max_daily_sector_trades)
    if traded.empty:
        return {
            "rows": out,
            "traded": traded,
            "portfolio_returns": pd.Series(dtype=float),
            "accuracy": 0,
            "coverage": 0,
            "signal_coverage": signal_count / len(out) * 100,
            "avg_return": 0,
            "profit_factor": 0,
            "portfolio_profit_factor": 0,
            "max_drawdown": 0,
            "brier": brier_score_loss(out["target"], out["prob_up"]),
            "sharpe": 0,
            "annualized_sharpe": 0,
            "psr": 0,
            "dsr": 0,
            "up_threshold": up_threshold,
            "down_threshold": down_threshold,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_position_size": max_position_size,
            "max_portfolio_exposure": max_portfolio_exposure,
            "max_sector_exposure": max_sector_exposure,
            "daily_stop": daily_stop,
            "equity_breaker": equity_breaker,
            "strict_filter": strict_filter,
            "execution_mode": execution_mode,
            "stress_slippage_bps": stress_slippage_bps,
            "risk_mode": risk_mode,
            "max_daily_trades": max_daily_trades,
            "max_daily_sector_trades": max_daily_sector_trades,
            "max_consecutive_loss_days": max_consecutive_loss_days,
            "loss_cooldown_days": loss_cooldown_days,
            "breaker_triggered": False,
            "breaker_trigger_date": None,
            "days_after_breaker": 0,
            "cooldown_trigger_count": 0,
            "cooldown_days_used": 0,
        }
    portfolio_returns = portfolio_return_series(
        traded,
        fee_bps,
        max_portfolio_exposure,
        max_sector_exposure,
        daily_stop,
        equity_breaker,
        max_consecutive_loss_days,
        loss_cooldown_days,
    )
    equity = (1 + portfolio_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    gains = traded.loc[traded["strategy_return"] > 0, "strategy_return"].sum()
    losses = -traded.loc[traded["strategy_return"] < 0, "strategy_return"].sum()
    portfolio_gains = portfolio_returns[portfolio_returns > 0].sum()
    portfolio_losses = -portfolio_returns[portfolio_returns < 0].sum()
    breaker_triggered = bool(portfolio_returns.attrs.get("breaker_triggered", False))
    breaker_trigger_date = portfolio_returns.attrs.get("breaker_trigger_date")
    days_after_breaker = int(portfolio_returns.attrs.get("days_after_breaker", 0))
    cooldown_trigger_count = int(portfolio_returns.attrs.get("cooldown_trigger_count", 0))
    cooldown_days_used = int(portfolio_returns.attrs.get("cooldown_days_used", 0))
    sharpe, annual_sr, psr, dsr = portfolio_risk_metrics(portfolio_returns, trials=trials)

    return {
        "rows": out,
        "traded": traded,
        "portfolio_returns": portfolio_returns,
        "calibration_bins": [] if fast else calibration_bins_report(out),
        "monte_carlo": {} if fast else monte_carlo_trade_shuffle(traded),
        "temporal_stability": [] if fast else temporal_stability_audit(traded),
        "ticker_dependency": [] if fast else ticker_dependency_audit(traded),
        "accuracy": traded["correct"].mean() * 100,
        "coverage": len(traded) / len(out) * 100,
        "signal_coverage": signal_count / len(out) * 100,
        "avg_return": traded["strategy_return"].mean() * 100,
        "profit_factor": gains / losses if losses > 0 else np.inf,
        "portfolio_profit_factor": portfolio_gains / portfolio_losses if portfolio_losses > 0 else np.inf,
        "max_drawdown": drawdown.min() * 100 if len(drawdown) else 0,
        "portfolio_avg_daily_return": portfolio_returns.mean() * 100 if len(portfolio_returns) else 0,
        "portfolio_total_return": (equity.iloc[-1] - 1) * 100 if len(equity) else 0,
        "breaker_triggered": breaker_triggered,
        "breaker_trigger_date": breaker_trigger_date,
        "days_after_breaker": days_after_breaker,
        "cooldown_trigger_count": cooldown_trigger_count,
        "cooldown_days_used": cooldown_days_used,
        "failure_reason": "historic regime failure" if breaker_triggered and days_after_breaker > 100 else "",
        "brier": brier_score_loss(out["target"], out["prob_up"]),
        "sharpe": sharpe,
        "annualized_sharpe": annual_sr,
        "psr": psr,
        "dsr": dsr,
        "dsr_trials": trials,
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_position_size": max_position_size,
        "max_portfolio_exposure": max_portfolio_exposure,
        "max_sector_exposure": max_sector_exposure,
        "daily_stop": daily_stop,
        "equity_breaker": equity_breaker,
        "strict_filter": strict_filter,
        "execution_mode": execution_mode,
        "stress_slippage_bps": stress_slippage_bps,
        "risk_mode": risk_mode,
        "max_daily_trades": max_daily_trades,
        "max_daily_sector_trades": max_daily_sector_trades,
        "max_consecutive_loss_days": max_consecutive_loss_days,
        "loss_cooldown_days": loss_cooldown_days,
    }


def optimise_thresholds(
    rows,
    fee_bps,
    min_coverage,
    stop_loss,
    take_profit,
    max_position_size=DEFAULT_MAX_POSITION_SIZE,
    max_portfolio_exposure=DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
    daily_stop=DEFAULT_DAILY_STOP,
    equity_breaker=DEFAULT_EQUITY_BREAKER,
    strict_filter=False,
    execution_mode="stop_first",
    stress_slippage_bps=0,
    risk_mode="fixed",
    meta_model=None,
    meta_threshold=None,
):
    best = None
    # Symmetric search grid: both up and down thresholds are searched over the
    # same distance-from-0.5 range (0.04 to 0.30 edge required), instead of the
    # old grid which searched a much wider, easier range for longs (0.54-0.80)
    # than for shorts (0.20-0.46 => up to 0.30 edge required either way, but the
    # up side started allowing signals at just +0.04 edge while down required at
    # least the same). This removes the structural long-side bias in threshold
    # tuning, so the optimizer is not simply defaulting to "trade longs more."
    edge_values = np.arange(0.04, 0.31, 0.02)
    up_values = np.round(0.50 + edge_values, 4)
    down_values = np.round(0.50 - edge_values, 4)

    # The grid only needs the threshold selection metrics (accuracy, PF, DD,
    # coverage, brier); the expensive report fields (monte carlo shuffle,
    # calibration bins, temporal/ticker audits) are re-run once on the winner
    # below, so ~91 grid evals per fold skip that overhead.
    def _grid_eval(up, down):
        return evaluate_predictions(
            rows,
            fee_bps,
            float(up),
            float(down),
            stop_loss,
            take_profit,
            max_position_size,
            max_portfolio_exposure,
            max_sector_exposure,
            daily_stop,
            equity_breaker,
            strict_filter,
            execution_mode,
            stress_slippage_bps,
            risk_mode,
            meta_model=meta_model,
            meta_threshold=meta_threshold,
            fast=True,
        )

    def _full_eval(up, down):
        return evaluate_predictions(
            rows,
            fee_bps,
            float(up),
            float(down),
            stop_loss,
            take_profit,
            max_position_size,
            max_portfolio_exposure,
            max_sector_exposure,
            daily_stop,
            equity_breaker,
            strict_filter,
            execution_mode,
            stress_slippage_bps,
            risk_mode,
            meta_model=meta_model,
            meta_threshold=meta_threshold,
        )

    for up_threshold in up_values:
        for down_threshold in down_values:
            if down_threshold >= up_threshold:
                continue
            metrics = _grid_eval(up_threshold, down_threshold)
            if not metrics or metrics["coverage"] < min_coverage:
                continue
            pf = metrics["profit_factor"]
            score = (
                metrics["accuracy"] * 1.2
                + min(pf, 3) * 7
                + metrics["avg_return"] * 3
                + metrics["coverage"] * 0.05
                + metrics["max_drawdown"] * 0.45
                - metrics["brier"] * 10
            )
            if best is None or score > best[0]:
                best = (score, metrics)

    if best is not None:
        return _full_eval(best[1]["up_threshold"], best[1]["down_threshold"])

    # If the requested coverage is too strict, fall back to the best non-empty
    # threshold pair instead of silently keeping a weak manual threshold.
    for up_threshold in up_values:
        for down_threshold in down_values:
            if down_threshold >= up_threshold:
                continue
            metrics = _grid_eval(up_threshold, down_threshold)
            if metrics and metrics["coverage"] > 0:
                score = metrics["accuracy"] + min(metrics["profit_factor"], 3) * 5 + metrics["max_drawdown"] * 0.45
                if best is None or score > best[0]:
                    best = (score, metrics)
    if best is not None:
        return _full_eval(best[1]["up_threshold"], best[1]["down_threshold"])
    return evaluate_predictions(
        rows,
        fee_bps,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_position_size=max_position_size,
        max_portfolio_exposure=max_portfolio_exposure,
        max_sector_exposure=max_sector_exposure,
        daily_stop=daily_stop,
        equity_breaker=equity_breaker,
        strict_filter=strict_filter,
        execution_mode=execution_mode,
        stress_slippage_bps=stress_slippage_bps,
        risk_mode=risk_mode,
        meta_model=meta_model,
        meta_threshold=meta_threshold,
    )


def prediction_rows_from_model(model_name, model, train_data, test_data, fold_label,
                               horizon_days=DEFAULT_HORIZON_DAYS, purge=False, embargo_days=0):
    if train_data.empty or test_data.empty or train_data["target"].nunique() < 2:
        return []

    dates = train_data["date"].to_numpy() if "date" in train_data.columns else None
    fitted = calibrate_final_model(
        clone(model),
        train_data[ACTIVE_FEATURE_COLUMNS],
        train_data["target"],
        dates=dates,
        horizon=horizon_days if purge else 0,
        embargo=embargo_days if purge else 0,
    )
    raw_probs = fitted.predict_proba(test_data[ACTIVE_FEATURE_COLUMNS])[:, 1]
    rows = []
    for i, raw_prob in enumerate(raw_probs):
        row = test_data.iloc[i]
        adjusted_prob = regime_probability_overlay(float(raw_prob), row)
        rows.append(
            {
                "fold": fold_label,
                "date": row["date"],
                "ticker": row["ticker"],
                "target": int(row["target"]),
                "future_return": float(row["future_return"]),
                "future_max_return": float(row["future_max_return"]),
                "future_min_return": float(row["future_min_return"]),
                "raw_prob_up": float(raw_prob),
                "prob_up": adjusted_prob,
                "signal": 0,
                "model": model_name,
                "whale_footprint_score": float(row.get("whale_footprint_score", 0)),
                "risk_on_score": float(row.get("risk_on_score", 0.5)),
                "trend_score": float(row.get("trend_score", 0.5)),
                "volatility_score": float(row.get("volatility_score", 0.5)),
                "market_breadth_score": float(row.get("market_breadth_score", 0.5)),
                "credit_stress_score": float(row.get("credit_stress_score", 0.5)),
                "relative_strength_20d": float(row.get("relative_strength_20d", 0)),
                "atr_pct": float(row.get("atr_pct", 0)),
                "setup_candidate": int(row.get("setup_candidate", 1)),
                "pocket_pivot": float(row.get("pocket_pivot", 0)),
                "quiet_accumulation": float(row.get("quiet_accumulation", 0)),
                "quiet_distribution": float(row.get("quiet_distribution", 0)),
                "vwap_reclaim": float(row.get("vwap_reclaim", 0)),
                "vwap_rejection": float(row.get("vwap_rejection", 0)),
                "breakout_volume_20": float(row.get("breakout_volume_20", 0)),
                "breakdown_volume_20": float(row.get("breakdown_volume_20", 0)),
            }
        )
    return rows


def final_test_split(data, final_fraction=DEFAULT_FINAL_TEST_FRACTION):
    ordered = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(ordered["date"]).unique()).sort_values().reset_index(drop=True)
    if len(dates) < 30 or final_fraction <= 0:
        return ordered, ordered.iloc[0:0].copy(), None

    split_at = int(len(dates) * (1 - final_fraction))
    split_at = max(1, min(len(dates) - 1, split_at))
    cutoff = dates.iloc[split_at - 1]
    tuning = ordered[pd.to_datetime(ordered["date"]) <= cutoff].copy()
    final = ordered[pd.to_datetime(ordered["date"]) > cutoff].copy()
    if len(tuning) < 100 or final.empty or tuning["target"].nunique() < 2:
        return ordered, ordered.iloc[0:0].copy(), None
    return tuning, final, cutoff


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Chronological folds with purging + embargo (AFML Ch.7).

    A training sample at time t is assumed to carry a label that spans the next
    `horizon` observations; it is purged from a fold when its label interval
    overlaps the fold's test window. An optional `embargo` additionally drops
    training observations near the boundary to mitigate serial correlation.

    When `dates` is provided (array aligned to the rows), positions are computed
    on the unique sorted trading-date axis so panel data (multiple tickers per
    date) purges correctly. Otherwise raw row positions are used.
    """

    def __init__(self, n_splits=5, horizon=20, embargo=0, dates=None):
        self.n_splits = n_splits
        self.horizon = int(horizon)
        self.embargo = int(embargo)
        self.dates = None if dates is None else np.asarray(dates)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        if n < 2:
            return
        if self.dates is not None and len(self.dates) == n:
            unique_dates = np.unique(self.dates)
            pos_map = {d: i for i, d in enumerate(unique_dates)}
            positions = np.array([pos_map[d] for d in self.dates], dtype=np.int64)
            n_axis = len(unique_dates)
        else:
            positions = np.arange(n)
            n_axis = n

        bounds = np.linspace(0, n_axis, self.n_splits + 1).astype(int)
        for i in range(self.n_splits):
            test_start, test_end = int(bounds[i]), int(bounds[i + 1])
            if i == self.n_splits - 1:
                test_end = n_axis
            if test_end - test_start < 1:
                continue
            keep = (
                (positions + self.horizon < test_start)
                & (positions < test_start - self.embargo)
                & (positions < test_start)
            )
            train_idx = np.where(keep)[0]
            test_idx = np.where((positions >= test_start) & (positions < test_end))[0]
            if len(train_idx) >= 2 and len(test_idx) >= 1:
                yield train_idx, test_idx


def walk_forward_folds(data, n_splits, horizon_days, embargo_days=0):
    """Yield (train_idx, test_idx) purged chronological folds over a panel.

    Positions are defined on the unique sorted trading-date axis so rows for all
    tickers on a given date are treated as one time step. A training row at date
    position p is dropped when p + horizon_days reaches the test fold (AFML
    Ch.7 purging), plus an optional embargo.
    """
    dates = pd.Series(pd.to_datetime(data["date"].unique())).sort_values().reset_index(drop=True)
    n_axis = len(dates)
    if n_axis < 2:
        return
    bounds = np.linspace(0, n_axis, n_splits + 1).astype(int)
    pos_map = {d: i for i, d in enumerate(dates)}
    positions = data["date"].map(pos_map).to_numpy(dtype=np.int64)
    idx = np.arange(len(data))
    for i in range(n_splits):
        test_start, test_end = int(bounds[i]), int(bounds[i + 1])
        if i == n_splits - 1:
            test_end = n_axis
        if test_end - test_start < 1:
            continue
        keep = (
            (positions + horizon_days < test_start)
            & (positions < test_start - embargo_days)
            & (positions < test_start)
        )
        train_idx = idx[keep]
        test_idx = idx[(positions >= test_start) & (positions < test_end)]
        if len(train_idx) >= 2 and len(test_idx) >= 1:
            yield train_idx, test_idx


def purge_labels_overlapping(data, split_date, horizon_days, embargo_days=0):
    """Drop rows whose label window overlaps `split_date` or later.

    Used to keep the model that scores the untouched final test from training on
    labels that extend into the final window. `horizon_days` is in trading days,
    converted with a business-day offset.
    """
    if data is None or data.empty or (horizon_days <= 0 and embargo_days <= 0):
        return data
    split = pd.Timestamp(split_date)
    keep = pd.Series(True, index=data.index)
    if horizon_days > 0:
        keep &= pd.to_datetime(data["date"]) < (split - pd.offsets.BDay(horizon_days))
    if embargo_days > 0:
        keep &= pd.to_datetime(data["date"]) < (split - pd.offsets.BDay(embargo_days))
    return data[keep]


def walk_forward_backtest(
    data,
    model_name,
    model,
    up_threshold,
    down_threshold,
    fee_bps,
    min_coverage,
    stop_loss,
    take_profit,
    max_position_size=DEFAULT_MAX_POSITION_SIZE,
    max_portfolio_exposure=DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
    daily_stop=DEFAULT_DAILY_STOP,
    equity_breaker=DEFAULT_EQUITY_BREAKER,
    strict_filter=False,
    execution_mode="stop_first",
    stress_slippage_bps=0,
    final_test_fraction=DEFAULT_FINAL_TEST_FRACTION,
    risk_mode="fixed",
    horizon_days=DEFAULT_HORIZON_DAYS,
    purge=PURGE_CV_ENABLED,
    embargo_days=PURGE_EMBARGO_DAYS,
):
    tuning_data, final_data, final_cutoff = final_test_split(data, final_test_fraction)
    n_splits = min(6, max(2, len(tuning_data) // 250))
    if len(tuning_data) <= n_splits:
        n_splits = max(2, min(3, len(tuning_data) - 1))
    X = tuning_data[ACTIVE_FEATURE_COLUMNS]
    y = tuning_data["target"]
    if purge and horizon_days and horizon_days > 0:
        folds = walk_forward_folds(tuning_data, n_splits, horizon_days, embargo_days)
    else:
        folds = TimeSeriesSplit(n_splits=n_splits).split(X, y)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train = tuning_data.iloc[train_idx]
        test = tuning_data.iloc[test_idx]

        rows.extend(prediction_rows_from_model(model_name, model, train, test, fold, horizon_days, purge, embargo_days))

    manual_metrics = evaluate_predictions(
        rows,
        fee_bps,
        up_threshold,
        down_threshold,
        stop_loss,
        take_profit,
        max_position_size,
        max_portfolio_exposure,
        max_sector_exposure,
        daily_stop,
        equity_breaker,
        strict_filter,
        execution_mode,
        stress_slippage_bps,
        risk_mode,
    )
    tuned_metrics = optimise_thresholds(
        rows,
        fee_bps,
        min_coverage,
        stop_loss,
        take_profit,
        max_position_size,
        max_portfolio_exposure,
        max_sector_exposure,
        daily_stop,
        equity_breaker,
        strict_filter,
        execution_mode,
        stress_slippage_bps,
        risk_mode,
    )
    final_train = tuning_data
    if purge and final_cutoff is not None and horizon_days and horizon_days > 0:
        final_train = purge_labels_overlapping(tuning_data, final_cutoff, horizon_days, embargo_days)
    final_rows = prediction_rows_from_model(model_name, model, final_train, final_data, "final",
                                            horizon_days, purge, embargo_days)
    final_metrics = evaluate_predictions(
        final_rows,
        fee_bps,
        tuned_metrics.get("up_threshold"),
        tuned_metrics.get("down_threshold"),
        stop_loss,
        take_profit,
        max_position_size,
        max_portfolio_exposure,
        max_sector_exposure,
        daily_stop,
        equity_breaker,
        strict_filter,
        execution_mode,
        stress_slippage_bps,
        risk_mode,
    ) if final_rows else {}
    tuned_metrics["manual_metrics"] = {
        "accuracy": manual_metrics["accuracy"],
        "coverage": manual_metrics["coverage"],
        "avg_return": manual_metrics["avg_return"],
        "profit_factor": manual_metrics["profit_factor"],
        "portfolio_profit_factor": manual_metrics.get("portfolio_profit_factor", 0),
        "max_drawdown": manual_metrics["max_drawdown"],
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
    }
    tuned_metrics["final_test_metrics"] = final_metrics
    tuned_metrics["final_test_cutoff"] = final_cutoff
    tuned_metrics["feature_drift"] = feature_drift_audit(tuning_data, final_data, ACTIVE_FEATURE_COLUMNS)
    tuned_metrics["leakage_correlation"] = leakage_correlation_audit(tuning_data, ACTIVE_FEATURE_COLUMNS)
    tuned_metrics["feature_redundancy"] = feature_redundancy_audit(tuning_data, FEATURE_COLUMNS)
    tuned_metrics["_oos_rows"] = rows
    tuned_metrics["_final_rows"] = final_rows
    return tuned_metrics


ALLOCATOR_MODEL_NAMES = [
    name for name, model in candidate_models().items()
    if name != "ensemble"
]
META_FEATURE_COLUMNS = [
    "prob_edge",
    "agreement",
    "whale_alignment",
    "regime_alignment",
    "whale_footprint_score",
    "risk_on_score",
    "trend_score",
    "volatility_score",
    "market_breadth_score",
    "credit_stress_score",
    "relative_strength_20d",
    "atr_pct",
    "setup_candidate",
]
ALLOCATOR_SIGMA_SCALE = 0.15
ALLOCATOR_SHRINK_STRENGTH = 0.60


def fit_allocator_weights(results, oos_dfs):
    """Derive per-model capital weights from out-of-sample reliability.

    base_w = exp(-brier) * (1 + 0.5 * min(PF, 3)) * (0.5 + accuracy/100)
    Regime multipliers adjust a model's vote up/down depending on how it
    performed out-of-sample inside the currently active regime.
    """
    names = []
    base_weights = {}
    briers = {}
    for name in ALLOCATOR_MODEL_NAMES:
        metrics = results.get(name)
        df = oos_dfs.get(name)
        if not metrics or df is None or df.empty:
            continue
        brier = float(metrics.get("brier", 0.5))
        pf = metrics.get("profit_factor", 0)
        if pf == np.inf:
            pf = 3.0
        acc = float(metrics.get("accuracy", 50))
        base_w = np.exp(-brier) * (1 + 0.5 * min(pf, 3)) * (0.5 + acc / 100)
        names.append(name)
        base_weights[name] = float(base_w)
        briers[name] = brier

    if not names:
        return {}

    total = sum(base_weights.values())
    capital_shares = {name: base_weights[name] / total for name in names}

    regime_weights = {}
    regime_multipliers = {}
    for name in names:
        df = oos_dfs[name].copy()
        df["active_regime"] = df.apply(market_regime_label, axis=1)
        rw = {}
        for regime, group in df.groupby("active_regime"):
            if len(group) < 30 or group["target"].nunique() < 2:
                continue
            rb = brier_score_loss(group["target"], group["prob_up"])
            r_acc = float((group["prob_up"] >= 0.5).astype(int).eq(group["target"]).mean() * 100)
            rw[regime] = float(np.exp(-rb) * (0.5 + r_acc / 100))
        regime_weights[name] = rw
        mean_rw = float(np.mean(list(rw.values()))) if rw else base_weights[name]
        regime_multipliers[name] = {r: w / mean_rw for r, w in rw.items()}

    return {
        "model_names": names,
        "base_weights": base_weights,
        "capital_shares": capital_shares,
        "briers": briers,
        "regime_weights": regime_weights,
        "regime_multipliers": regime_multipliers,
    }


def allocator_regime_multiplier(allocator, name, regime):
    return (allocator.get("regime_multipliers", {}).get(name, {}) or {}).get(regime, 1.0)


def model_for_regime(payload, name, regime):
    specialists = (payload.get("regime_models_by_model") or {}).get(name) or {}
    model = specialists.get(regime)
    if model is not None:
        return model, "specialist"
    all_models = payload.get("all_models") or {}
    fallback = all_models.get(name) or payload.get("model")
    return fallback, "general"


def allocate_row_probabilities(out, allocator):
    names = [n for n in allocator.get("model_names", []) if f"prob_{n}" in out.columns]
    if not names:
        return out
    base_w = allocator["base_weights"]
    regime_mult = allocator.get("regime_multipliers", {})
    out["active_regime"] = out.apply(market_regime_label, axis=1)
    weights = np.zeros((len(out), len(names)))
    for j, name in enumerate(names):
        mult = out["active_regime"].map(lambda r: allocator_regime_multiplier(allocator, name, r))
        weights[:, j] = base_w.get(name, 1.0) * mult.to_numpy()
    probs = out[[f"prob_{n}" for n in names]].to_numpy()
    wsum = weights.sum(axis=1, keepdims=True)
    wsum[wsum <= 0] = 1.0
    p_blend = (weights * probs).sum(axis=1) / wsum[:, 0]
    centered = (probs - p_blend[:, None]) ** 2
    sigma = np.sqrt((weights * centered).sum(axis=1) / wsum[:, 0])
    agreement = np.clip(1 - sigma / ALLOCATOR_SIGMA_SCALE, 0, 1)
    shrunk = p_blend + (0.5 - p_blend) * (1 - agreement) * ALLOCATOR_SHRINK_STRENGTH
    out["allocated_prob_up"] = shrunk
    out["prob_up"] = shrunk
    out["prob_sigma"] = sigma
    out["agreement"] = agreement
    return out


def build_allocated_rows(oos_dfs, allocator):
    names = [n for n in allocator.get("model_names", []) if oos_dfs.get(n) is not None and not oos_dfs[n].empty]
    if not names:
        return pd.DataFrame()
    key = ["fold", "date", "ticker"]
    out = oos_dfs[names[0]].copy()
    for name in names:
        df = oos_dfs[name]
        cols = [c for c in ["prob_up", "raw_prob_up"] if c in df.columns]
        right = df[key + cols].rename(
            columns={"prob_up": f"prob_{name}", "raw_prob_up": f"raw_prob_{name}"}
        )
        out = out.merge(right, on=key, how="inner")
    out["prob_up"] = out.get(f"prob_{names[0]}", 0.5)
    return out


def allocated_live_probability(payload, latest_row, X_latest):
    """Blend all trained models for a single live feature row."""
    allocator = payload.get("allocator") or {}
    all_models = payload.get("all_models") or {}
    if not allocator or not all_models:
        model, regime, route = select_payload_model(payload, latest_row)
        raw = float(model.predict_proba(X_latest)[:, 1][0])
        prob = regime_probability_overlay(raw, latest_row)
        return {
            "prob_up": prob,
            "raw_prob_up": raw,
            "sigma": 0.0,
            "agreement": 1.0,
            "regime": regime,
            "model_route": route,
            "model_probs": {},
            "capital_shares": {},
        }

    regime = market_regime_label(latest_row)
    names = [n for n in allocator.get("model_names", []) if n in all_models]
    if not names:
        model, regime, route = select_payload_model(payload, latest_row)
        raw = float(model.predict_proba(X_latest)[:, 1][0])
        return {
            "prob_up": regime_probability_overlay(raw, latest_row),
            "raw_prob_up": raw,
            "sigma": 0.0,
            "agreement": 1.0,
            "regime": regime,
            "model_route": route,
            "model_probs": {},
            "capital_shares": {},
        }

    base_w = allocator["base_weights"]
    probs = {}
    routes = {}
    for n in names:
        m, route = model_for_regime(payload, n, regime)
        probs[n] = float(m.predict_proba(X_latest)[:, 1][0])
        routes[n] = route

    w = np.array([base_w.get(n, 1.0) * allocator_regime_multiplier(allocator, n, regime) for n in names])
    w[w <= 0] = 1e-6
    p = np.array([probs[n] for n in names])
    P = float((w @ p) / w.sum())
    sigma = float(np.sqrt((w * (p - P) ** 2).sum() / w.sum()))
    agreement = float(np.clip(1 - sigma / ALLOCATOR_SIGMA_SCALE, 0, 1))
    shrunk = float(P + (0.5 - P) * (1 - agreement) * ALLOCATOR_SHRINK_STRENGTH)
    caps = w / w.sum()
    return {
        "prob_up": shrunk,
        "raw_prob_up": P,
        "sigma": sigma,
        "agreement": agreement,
        "regime": regime,
        "model_route": "allocator",
        "model_probs": {n: float(x) for n, x in zip(names, p)},
        "capital_shares": {n: float(x) for n, x in zip(names, caps)},
    }


def meta_feature_row(row, agreement=1.0):
    signal = int(row.get("signal", 0))
    prob_up = float(row.get("prob_up", 0.5))
    whale = float(row.get("whale_footprint_score", row.get("whale_score", 0)))
    risk_on = float(row.get("risk_on_score", 0.5))
    edge = prob_up - 0.5 if signal == 1 else (0.5 - prob_up if signal == -1 else 0)
    whale_align = whale if signal == 1 else (-whale if signal == -1 else 0)
    regime_align = risk_on if signal == 1 else (1 - risk_on if signal == -1 else 0.5)
    return {
        "prob_edge": edge,
        "agreement": float(agreement),
        "whale_alignment": whale_align,
        "regime_alignment": regime_align,
        "whale_footprint_score": whale,
        "risk_on_score": risk_on,
        "trend_score": float(row.get("trend_score", 0.5)),
        "volatility_score": float(row.get("volatility_score", 0.5)),
        "market_breadth_score": float(row.get("market_breadth_score", 0.5)),
        "credit_stress_score": float(row.get("credit_stress_score", 0.5)),
        "relative_strength_20d": float(row.get("relative_strength_20d", 0)),
        "atr_pct": float(row.get("atr_pct", 0)),
        "setup_candidate": int(bool(row.get("setup_candidate", 1))),
    }


def train_meta_model(traded, fee_bps, stop_loss, take_profit, risk_mode="fixed", stress_slippage_bps=0):
    """Meta-labeling: learn which signaled trades actually turn profitable.

    Uses the OOS traded rows (signals already applied). Label = risk-managed
    trade outcome > 0. Returns (meta_model, threshold) where threshold is tuned
    on the same rows to maximize accuracy*edge on profitable trades.
    """
    if traded is None or traded.empty:
        return None, None
    df = traded.copy()
    if "agreement" not in df:
        df["agreement"] = 1.0
    features = []
    labels = []
    for _, row in df.iterrows():
        if row["signal"] == 0:
            continue
        labels.append(
            1 if risk_managed_return(row, stop_loss, take_profit, "stop_first", stress_slippage_bps, risk_mode) > 0 else 0
        )
        features.append(meta_feature_row(row))
    if len(features) < 40 or len(set(labels)) < 2:
        return None, None
    X = pd.DataFrame(features)[META_FEATURE_COLUMNS].fillna(0)
    y = np.array(labels)
    base_model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=12,
        l2_regularization=0.3,
        random_state=42,
    )
    meta_model = calibrate_final_model(base_model, X, y)

    probs = meta_model.predict_proba(X)[:, 1]
    min_keep = max(META_MIN_KEEP_ROWS, int(META_MIN_KEEP_FRACTION * len(y)))
    best_threshold, best_score = None, -1e9
    for threshold in np.arange(0.48, 0.62, 0.01):
        kept = probs >= threshold
        if kept.sum() < min_keep:
            continue
        gain = y[kept].sum()
        accuracy = float(y[kept].mean() * 100)
        coverage = float(kept.mean() * 100)
        score = accuracy + min(gain, len(y) * 0.5) * 0.1 + coverage * 0.05
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    if best_threshold is None:
        # No grid threshold keeps enough trades: fall back to the floor boundary
        # so the gate never collapses coverage to a handful of trades.
        floor_prob = np.quantile(probs, 1.0 - META_MIN_KEEP_FRACTION)
        best_threshold = min(float(floor_prob), 0.52)
    meta_model.meta_threshold_ = best_threshold
    return meta_model, best_threshold


def apply_meta_gate_and_size(traded, meta_model, meta_threshold):
    if traded is None or traded.empty:
        return traded
    if meta_model is None:
        return traded
    df = traded.copy()
    if "agreement" not in df:
        df["agreement"] = 1.0
    meta_features = pd.DataFrame([meta_feature_row(row) for _, row in df.iterrows()])[META_FEATURE_COLUMNS].fillna(0)
    df["meta_prob"] = meta_model.predict_proba(meta_features)[:, 1]
    kept = df[df["meta_prob"] >= meta_threshold].copy()
    if kept.empty:
        return kept.iloc[0:0].copy()
    # Lopez de Prado meta-labeling (AFML Ch.3): the bet size is the meta model's
    # estimate of the probability that the trade profits, applied to the base
    # position size. This replaces the old discrete edge bucketing, which zeroed
    # out every trade near the threshold.
    kept["position_size"] = kept["position_size"] * kept["meta_prob"]
    return kept


def print_allocator_report(allocator, meta_info=None):
    if not allocator:
        return
    section("DYNAMIC MODEL ALLOCATOR")
    rows = []
    for name in allocator["model_names"]:
        rows.append(
            [
                name,
                f"{allocator['capital_shares'][name] * 100:.1f}%",
                f"{allocator['briers'][name]:.3f}",
            ]
        )
    print(tabulate(rows, headers=["Model", "Capital Share", "OOS Brier"], tablefmt="simple"))
    if meta_info:
        meta_model, meta_threshold = meta_info
        if meta_model is not None:
            print(f"  Meta-label gate: keep trades where meta_prob >= {meta_threshold:.2f}")


def evaluate_allocated_predictions(
    oos_dfs,
    allocator,
    fee_bps,
    min_coverage,
    stop_loss,
    take_profit,
    up_threshold=None,
    down_threshold=None,
    max_position_size=DEFAULT_MAX_POSITION_SIZE,
    max_portfolio_exposure=DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
    daily_stop=DEFAULT_DAILY_STOP,
    equity_breaker=DEFAULT_EQUITY_BREAKER,
    strict_filter=False,
    execution_mode="stop_first",
    stress_slippage_bps=0,
    risk_mode="fixed",
    meta_model=None,
    meta_threshold=None,
    final_dfs=None,
    final_cutoff=None,
):
    rows_df = build_allocated_rows(oos_dfs, allocator)
    if rows_df.empty:
        return {}
    rows_df = allocate_row_probabilities(rows_df, allocator)
    rows = rows_df.to_dict("records")

    manual_metrics = evaluate_predictions(
        rows, fee_bps, up_threshold, down_threshold, stop_loss, take_profit,
        max_position_size, max_portfolio_exposure, max_sector_exposure,
        daily_stop, equity_breaker, strict_filter, execution_mode,
        stress_slippage_bps, risk_mode,
        meta_model=meta_model, meta_threshold=meta_threshold,
    )
    tuned_metrics = optimise_thresholds(
        rows, fee_bps, min_coverage, stop_loss, take_profit,
        max_position_size, max_portfolio_exposure, max_sector_exposure,
        daily_stop, equity_breaker, strict_filter, execution_mode,
        stress_slippage_bps, risk_mode,
        meta_model=meta_model, meta_threshold=meta_threshold,
    )
    tuned_metrics["manual_metrics"] = {
        "accuracy": manual_metrics["accuracy"],
        "coverage": manual_metrics["coverage"],
        "avg_return": manual_metrics["avg_return"],
        "profit_factor": manual_metrics["profit_factor"],
        "portfolio_profit_factor": manual_metrics.get("portfolio_profit_factor", 0),
        "max_drawdown": manual_metrics["max_drawdown"],
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
    }

    final_metrics = {}
    if final_dfs:
        final_df = build_allocated_rows(final_dfs, allocator)
        if not final_df.empty:
            final_df = allocate_row_probabilities(final_df, allocator)
            final_metrics = evaluate_predictions(
                final_df.to_dict("records"), fee_bps,
                tuned_metrics.get("up_threshold"), tuned_metrics.get("down_threshold"),
                stop_loss, take_profit,
                max_position_size, max_portfolio_exposure, max_sector_exposure,
                daily_stop, equity_breaker, strict_filter, execution_mode,
                stress_slippage_bps, risk_mode,
                meta_model=meta_model, meta_threshold=meta_threshold,
            )
    tuned_metrics["final_test_metrics"] = final_metrics
    tuned_metrics["final_test_cutoff"] = final_cutoff
    tuned_metrics["allocator"] = allocator
    return tuned_metrics


def pick_best_model(results):
    ranked = []
    for name, metrics in results.items():
        if not metrics:
            continue
        final_metrics = metrics.get("final_test_metrics") or {}
        final_accuracy = final_metrics.get("accuracy", metrics.get("accuracy", 0))
        final_pf = final_metrics.get("profit_factor", 0)
        final_portfolio_pf = final_metrics.get("portfolio_profit_factor", 0)
        final_dd = final_metrics.get("max_drawdown", 0)
        portfolio_pf = metrics.get("portfolio_profit_factor", 0)
        paused_days = metrics.get("days_after_breaker", 0)
        cooldowns = metrics.get("cooldown_trigger_count", 0)
        score = (
            metrics["accuracy"] * 0.35
            + final_accuracy * 0.55
            + metrics["coverage"] * 0.02
            + min(metrics["profit_factor"], 3) * 8
            + min(final_pf, 3) * 10
            + min(portfolio_pf, 3) * 2
            + min(final_portfolio_pf, 3) * 6
            + metrics["avg_return"] * 4
            + metrics["max_drawdown"] * 0.25
            + final_dd * 0.50
            - metrics["brier"] * 15
            + (metrics.get("dsr", 0.0) - 0.5) * 8
            + (final_metrics.get("dsr", 0.0) - 0.5) * 6
            - paused_days * 0.01
            - cooldowns * 0.10
        )
        ranked.append((score, name, metrics))
    return max(ranked, key=lambda x: x[0]) if ranked else None


def model_is_deployable(metrics):
    """
    Determine whether a model is safe to promote from EXPERIMENTAL
    to DEPLOYABLE.

    Focus on profitability, robustness and out-of-sample performance,
    rather than raw classification accuracy.
    """

    final = metrics.get("final_test_metrics") or {}

    # Never deploy without an untouched final test.
    if not final:
        return False

    min_trades = globals().get("DEFAULT_MIN_FINAL_TRADES", 20)

    # Walk-forward validation
    walkforward_pass = (
        metrics.get("coverage", 0) >= 8
        and metrics.get("profit_factor", 0) >= 1.20
        and metrics.get("portfolio_profit_factor", 0) >= 2.00
        and metrics.get("avg_return", 0) > 0
        and metrics.get("max_drawdown", -100) > -20
    )

    # Untouched out-of-sample validation
    final_pass = (
        final.get("coverage", 0) >= 10
        and len(final.get("traded", [])) >= min_trades
        and final.get("profit_factor", 0) >= 1.30
        and final.get("portfolio_profit_factor", 0) >= 2.00
        and final.get("max_drawdown", -100) > -15
        and final.get("accuracy", 0) >= 55
        and final.get("dsr", 1.0) > MIN_DEPLOY_DSR
    )

    return walkforward_pass and final_pass


def model_status(metrics):
    return "DEPLOYABLE" if model_is_deployable(metrics) else "EXPERIMENTAL"


def calibrate_final_model(model, X, y, dates=None, horizon=0, embargo=0):
    """Fit a calibrated classifier using time-series aware CV.

    Isotonic calibration usually fixes RandomForest overconfidence better than sigmoid,
    but it needs enough rows per fold. If isotonic cannot fit safely, fall back to sigmoid,
    then finally to the raw model.

    When `horizon > 0`, folds are purged/embargoed (AFML Ch.7) so training labels
    do not overlap the validation windows; `dates` (aligned to X) enables correct
    purging for panel data.
    """
    try:
        n_rows = len(X)
        n_splits = 4 if n_rows >= 400 else 3 if n_rows >= 180 else 2
        if len(pd.Series(y).unique()) < 2 or n_rows < 80:
            model.fit(X, y)
            return model

        if horizon and horizon > 0:
            cv = PurgedTimeSeriesSplit(n_splits=n_splits, horizon=horizon, embargo=embargo, dates=dates)
        else:
            cv = TimeSeriesSplit(n_splits=n_splits)
        method = "isotonic" if n_rows >= 250 else "sigmoid"
        calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
        calibrated.fit(X, y)
        calibrated.calibration_method_ = method
        return calibrated
    except Exception:
        try:
            cv = TimeSeriesSplit(n_splits=2)
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=cv)
            calibrated.fit(X, y)
            calibrated.calibration_method_ = "sigmoid"
            return calibrated
        except Exception:
            model.fit(X, y)
            return model

def train_regime_models(base_model_name, data):
    regime_models = {}
    regime_info = {}
    base_model = candidate_models()[base_model_name]
    for regime, group in data.groupby("regime_label"):
        if len(group) < MIN_REGIME_MODEL_ROWS or group["target"].nunique() < 2:
            regime_info[regime] = {
                "rows": int(len(group)),
                "trained": False,
                "reason": "not enough two-class rows",
            }
            continue
        model = calibrate_final_model(clone(base_model), group[ACTIVE_FEATURE_COLUMNS], group["target"])
        regime_models[regime] = model
        regime_info[regime] = {
            "rows": int(len(group)),
            "bull_rate": round(float(group["target"].mean()), 4),
            "trained": True,
        }
    return regime_models, regime_info


def train_and_backtest(
    tickers,
    mem,
    period,
    horizon_days,
    return_threshold,
    training_mode,
    up_threshold,
    down_threshold,
    min_coverage,
    fee_bps,
    stop_loss,
    take_profit,
    max_position_size=DEFAULT_MAX_POSITION_SIZE,
    max_portfolio_exposure=DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
    daily_stop=DEFAULT_DAILY_STOP,
    equity_breaker=DEFAULT_EQUITY_BREAKER,
    labeling="fixed",
):
    section("BUILDING WALK-FORWARD DATASET")
    data = make_dataset(tickers, period, horizon_days, return_threshold, training_mode,
                        labeling=labeling, stop_loss=stop_loss, take_profit=take_profit)
    if data.empty:
        print(f"  {Fore.RED}No usable training rows. Try a longer period or different tickers.{Style.RESET_ALL}")
        return None
    if data["target"].nunique() < 2:
        print(f"  {Fore.RED}Training rows only contain one class. Try all-days mode, a longer period, or a lower move threshold.{Style.RESET_ALL}")
        return None

    print(f"  Rows: {len(data):,} | Dates: {data['date'].min().date()} -> {data['date'].max().date()}")
    print(f"  Bull labels: {data['target'].mean() * 100:.1f}% | Bear labels: {(1 - data['target'].mean()) * 100:.1f}%")
    print_phase_audit_report("PHASE 1 LEAKAGE CORRELATION AUDIT", leakage_correlation_audit(data, ACTIVE_FEATURE_COLUMNS), max_rows=8)
    print_phase_audit_report("PHASE 1 FEATURE REDUNDANCY AUDIT", feature_redundancy_audit(data, ACTIVE_FEATURE_COLUMNS), max_rows=8)

    section("WALK-FORWARD MODEL COMPARISON")
    results = {}
    oos_dfs = {}
    final_dfs = {}
    for name, model in candidate_models().items():
        try:
            metrics = walk_forward_backtest(
                data,
                name,
                model,
                up_threshold,
                down_threshold,
                fee_bps,
                min_coverage,
                stop_loss,
                take_profit,
                max_position_size,
                max_portfolio_exposure,
                max_sector_exposure,
                daily_stop,
                equity_breaker,
                False,
                horizon_days=horizon_days,
            )
        except Exception as exc:
            print(f"  {Fore.YELLOW}Model {name} failed: {exc}{Style.RESET_ALL}")
            continue
        results[name] = metrics
        oos_dfs[name] = pd.DataFrame(metrics.get("_oos_rows") or [])
        final_dfs[name] = pd.DataFrame(metrics.get("_final_rows") or [])
        print_model_metrics(name, metrics)

    if not results:
        print(f"  {Fore.RED}No model produced results.{Style.RESET_ALL}")
        return None

    allocator = fit_allocator_weights(results, oos_dfs)
    if not allocator:
        print(f"  {Fore.RED}Could not build allocator weights.{Style.RESET_ALL}")
        return None

    final_cutoff = next((metrics.get("final_test_cutoff") for metrics in results.values()), None)

    print_allocator_report(allocator)
    allocated_metrics = evaluate_allocated_predictions(
        oos_dfs, allocator, fee_bps, min_coverage, stop_loss, take_profit,
        up_threshold, down_threshold,
        max_position_size, max_portfolio_exposure, max_sector_exposure,
        daily_stop, equity_breaker, False,
        final_dfs=final_dfs, final_cutoff=final_cutoff,
    )
    if not allocated_metrics:
        print(f"  {Fore.RED}Allocated evaluation produced no results.{Style.RESET_ALL}")
        return None
    section("DYNAMIC MODEL ALLOCATOR")
    print_model_metrics("allocator", allocated_metrics)
    plain_allocated_metrics = allocated_metrics

    meta_model, meta_threshold = train_meta_model(
        allocated_metrics.get("traded"), fee_bps, stop_loss, take_profit
    )
    if meta_model is not None:
        print(f"  Meta-label gate threshold: {meta_threshold:.2f}")
        meta_allocated_metrics = evaluate_allocated_predictions(
            oos_dfs, allocator, fee_bps, min_coverage, stop_loss, take_profit,
            up_threshold, down_threshold,
            max_position_size, max_portfolio_exposure, max_sector_exposure,
            daily_stop, equity_breaker, False,
            meta_model=meta_model, meta_threshold=meta_threshold,
            final_dfs=final_dfs, final_cutoff=final_cutoff,
        )
        meta_coverage = meta_allocated_metrics.get("coverage", 0) if meta_allocated_metrics else 0
        plain_coverage = plain_allocated_metrics.get("coverage", 0)
        degenerate = (
            not meta_allocated_metrics
            or meta_coverage < META_ABS_MIN_COVERAGE
            or meta_coverage < META_FLOOR_RELATIVE * plain_coverage
        )
        if degenerate:
            print(
                f"  {Fore.YELLOW}Meta gate collapsed coverage to {meta_coverage:.1f}% "
                f"(plain allocator {plain_coverage:.1f}%); falling back to plain allocator."
                f"{Style.RESET_ALL}"
            )
            meta_model = None
            meta_threshold = None
            allocated_metrics = plain_allocated_metrics
        else:
            allocated_metrics = meta_allocated_metrics
            section("DYNAMIC MODEL ALLOCATOR + META GATE")
            print_model_metrics("allocator+meta", allocated_metrics)

    best_metrics = allocated_metrics
    best_name = "allocator"
    top_model_name = max(allocator["capital_shares"], key=allocator["capital_shares"].get)

    print_by_ticker(best_metrics["traded"])
    print_phase_audit_report("PHASE 1 TEMPORAL STABILITY AUDIT", best_metrics.get("temporal_stability", []), max_rows=10)
    print_phase_audit_report("PHASE 1 CALIBRATION AUDIT", best_metrics.get("calibration_bins", []), max_rows=10)
    print_phase_audit_report("PHASE 1 FEATURE DRIFT AUDIT", best_metrics.get("feature_drift", []), max_rows=10)
    print_phase_audit_report("PHASE 2 TICKER DEPENDENCY AUDIT", best_metrics.get("ticker_dependency", []), max_rows=10)
    mc = best_metrics.get("monte_carlo", {})
    if mc:
        section("PHASE 1 MONTE CARLO SHUFFLE AUDIT")
        print(tabulate([[k, v] for k, v in mc.items()], headers=["Metric", "Value"], tablefmt="simple"))
    print_feature_importance(top_model_name, candidate_models()[top_model_name], data)
    plot_equity(best_metrics.get("portfolio_returns"))

    status = model_status(best_metrics)
    if status != "DEPLOYABLE":
        print(
            f"  {Fore.YELLOW}No deployable model found for this run. "
            f"Saving as EXPERIMENTAL so live predictions still work with the current feature set."
            f"{Style.RESET_ALL}"
        )

    section("TRAINING FINAL ALLOCATOR MODELS")
    all_models = {}
    for name in ALLOCATOR_MODEL_NAMES:
        try:
            all_models[name] = calibrate_final_model(candidate_models()[name], data[ACTIVE_FEATURE_COLUMNS], data["target"])
        except Exception as exc:
            print(f"  {Fore.YELLOW}Final training failed for {name}: {exc}{Style.RESET_ALL}")
            continue
    final_model = all_models.get(top_model_name) or next(iter(all_models.values()))
    regime_models_by_model = {}
    regime_model_info = {}
    for name in all_models:
        rm, ri = train_regime_models(name, data)
        regime_models_by_model[name] = rm
        regime_model_info[name] = ri
    if regime_model_info:
        section("REGIME SPECIALIST MODELS")
        for name, ri in sorted(regime_model_info.items()):
            rows = []
            for regime, info in sorted(ri.items()):
                rows.append(
                    [
                        regime,
                        info.get("rows", 0),
                        "yes" if info.get("trained") else "no",
                        info.get("bull_rate", ""),
                        info.get("reason", ""),
                    ]
                )
            print(f"  {name}:")
            print(tabulate(rows, headers=["Regime", "Rows", "Trained", "Bull Rate", "Note"], tablefmt="simple"))
    payload = {
        "model": final_model,
        "model_type": "model_allocator",
        "general_model_name": top_model_name,
        "all_models": all_models,
        "allocator": allocator,
        "meta_model": meta_model,
        "meta_threshold": meta_threshold,
        "regime_models_by_model": regime_models_by_model,
        "regime_models": regime_models_by_model.get(top_model_name, {}),
        "regime_model_info": regime_model_info,
        "features": ACTIVE_FEATURE_COLUMNS,
        "feature_stats": {
            col: {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()) if float(data[col].std()) != 0 else 1.0,
            }
            for col in ACTIVE_FEATURE_COLUMNS
            if col in data.columns
        },
        "phase_audits": {
            "feature_drift": best_metrics.get("feature_drift", []),
            "leakage_correlation": best_metrics.get("leakage_correlation", []),
            "feature_redundancy": best_metrics.get("feature_redundancy", [])[:25],
            "monte_carlo": best_metrics.get("monte_carlo", {}),
            "temporal_stability": best_metrics.get("temporal_stability", []),
            "ticker_dependency": best_metrics.get("ticker_dependency", []),
        },
        "tickers": tickers,
        "period": period,
        "horizon_days": horizon_days,
        "return_threshold": return_threshold,
        "training_mode": training_mode,
        "prob_up": best_metrics["up_threshold"],
        "prob_down": best_metrics["down_threshold"],
        "min_coverage": min_coverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_position_size": max_position_size,
        "max_portfolio_exposure": max_portfolio_exposure,
        "max_sector_exposure": max_sector_exposure,
        "daily_stop": daily_stop,
        "equity_breaker": equity_breaker,
        "model_status": status,
        "metrics": {
            "sharpe": best_metrics.get("sharpe", 0),
            "annualized_sharpe": best_metrics.get("annualized_sharpe", 0),
            "psr": best_metrics.get("psr", 0),
            "dsr": best_metrics.get("dsr", 0),
            "dsr_trials": best_metrics.get("dsr_trials", DEFAULT_SELECTION_TRIALS),
        },
        "labeling": labeling,
        "trained_at": dt.datetime.now().isoformat(),
    }
    joblib.dump(payload, MODEL_FILE)
    print(f"  {Fore.GREEN}{status} allocator model saved -> {MODEL_FILE}{Style.RESET_ALL}")

    allocator_summary = {
        name: round(allocator["capital_shares"].get(name, 0), 4)
        for name in allocator["model_names"]
    }
    mem["best_model"] = {
        "name": best_name,
        "accuracy": round(best_metrics["accuracy"], 2),
        "coverage": round(best_metrics["coverage"], 2),
        "profit_factor": round(float(best_metrics["profit_factor"]), 3)
        if np.isfinite(best_metrics["profit_factor"])
        else "inf",
        "portfolio_profit_factor": round(float(best_metrics.get("portfolio_profit_factor", 0)), 3)
        if np.isfinite(best_metrics.get("portfolio_profit_factor", 0))
        else "inf",
        "avg_return": round(best_metrics["avg_return"], 3),
        "max_drawdown": round(best_metrics["max_drawdown"], 2),
        "brier": round(best_metrics["brier"], 4),
        "sharpe": round(best_metrics.get("sharpe", 0), 3),
        "psr": round(best_metrics.get("psr", 0), 3),
        "dsr": round(best_metrics.get("dsr", 0), 3),
        "final_accuracy": round((best_metrics.get("final_test_metrics") or {}).get("accuracy", 0), 2),
        "final_coverage": round((best_metrics.get("final_test_metrics") or {}).get("coverage", 0), 2),
        "final_profit_factor": round(float((best_metrics.get("final_test_metrics") or {}).get("profit_factor", 0)), 3)
        if np.isfinite((best_metrics.get("final_test_metrics") or {}).get("profit_factor", 0))
        else "inf",
        "final_portfolio_profit_factor": round(float((best_metrics.get("final_test_metrics") or {}).get("portfolio_profit_factor", 0)), 3)
        if np.isfinite((best_metrics.get("final_test_metrics") or {}).get("portfolio_profit_factor", 0))
        else "inf",
        "final_max_drawdown": round((best_metrics.get("final_test_metrics") or {}).get("max_drawdown", 0), 2),
        "prob_up": round(best_metrics["up_threshold"], 3),
        "prob_down": round(best_metrics["down_threshold"], 3),
        "stop_loss": round(best_metrics["stop_loss"], 3),
        "take_profit": round(best_metrics["take_profit"], 3),
        "max_position_size": round(max_position_size, 3),
        "max_portfolio_exposure": round(max_portfolio_exposure, 3),
        "max_sector_exposure": round(max_sector_exposure, 3),
        "daily_stop": round(daily_stop, 3),
        "equity_breaker": round(equity_breaker, 3),
        "model_status": status,
        "trained_at": dt.datetime.now().isoformat(),
        "allocator_weights": allocator_summary,
        "meta_threshold": round(meta_threshold, 3) if meta_threshold is not None else None,
    }
    mem["runs"].append(mem["best_model"])
    log_model_run_to_db(mem["best_model"], tickers)
    save_memory(mem)
    return payload


def print_model_metrics(name, metrics):
    if not metrics:
        print(f"  {name}: no metrics")
        return
    pf = metrics["profit_factor"]
    pf_text = "inf" if np.isinf(pf) else f"{pf:.2f}"
    portfolio_pf = metrics.get("portfolio_profit_factor", 0)
    portfolio_pf_text = "inf" if np.isinf(portfolio_pf) else f"{portfolio_pf:.2f}"
    thresholds = ""
    if metrics.get("up_threshold") is not None and metrics.get("down_threshold") is not None:
        thresholds = f"  thresholds={metrics['down_threshold']:.2f}/{metrics['up_threshold']:.2f}"
    risk = ""
    if metrics.get("stop_loss") is not None and metrics.get("take_profit") is not None:
        risk = f"  stop/tp={metrics['stop_loss']:.1%}/{metrics['take_profit']:.1%}"
    trade_limit = ""
    if metrics.get("max_daily_trades"):
        trade_limit = f"  daily_limit={metrics.get('max_daily_trades')}/{metrics.get('max_daily_sector_trades')}"
    cooldown = ""
    if metrics.get("cooldown_trigger_count", 0):
        cooldown = (
            f"  cooldowns={metrics.get('cooldown_trigger_count', 0)}"
            f"/{metrics.get('cooldown_days_used', 0)}d"
        )
    breaker = ""
    if metrics.get("breaker_triggered"):
        date = metrics.get("breaker_trigger_date")
        date_text = f" on {pd.to_datetime(date).date()}" if date is not None else ""
        breaker = f"  BREAKER{date_text}; paused_days={metrics.get('days_after_breaker', 0)}"
    failure = f"  failure={metrics['failure_reason']}" if metrics.get("failure_reason") else ""
    print(
        f"  {name:<24} "
        f"accuracy={metrics['accuracy']:5.1f}%  "
        f"coverage={metrics['coverage']:5.1f}%  "
        f"avg/trade={metrics['avg_return']:+6.2f}%  "
        f"profit_factor={pf_text:>5}  "
        f"portfolio_pf={portfolio_pf_text:>5}  "
        f"max_dd={metrics['max_drawdown']:6.1f}%  "
        f"brier={metrics['brier']:.3f}"
        f"{thresholds}"
        f"{risk}"
        f"{trade_limit}"
        f"{cooldown}"
        f"{breaker}"
        f"{failure}"
    )
    final_metrics = metrics.get("final_test_metrics") or {}
    if final_metrics:
        final_pf = final_metrics.get("profit_factor", 0)
        final_portfolio_pf = final_metrics.get("portfolio_profit_factor", 0)
        final_pf_text = "inf" if np.isinf(final_pf) else f"{final_pf:.2f}"
        final_portfolio_pf_text = "inf" if np.isinf(final_portfolio_pf) else f"{final_portfolio_pf:.2f}"
        cutoff = metrics.get("final_test_cutoff")
        cutoff_text = f" after {pd.to_datetime(cutoff).date()}" if cutoff is not None else ""
        print(
            f"    untouched final test{cutoff_text}: "
            f"accuracy={final_metrics['accuracy']:.1f}%  "
            f"coverage={final_metrics['coverage']:.1f}%  "
            f"pf={final_pf_text}  "
            f"portfolio_pf={final_portfolio_pf_text}  "
            f"max_dd={final_metrics['max_drawdown']:.1f}%  "
            f"breaker={bool(final_metrics.get('breaker_triggered', False))}  "
            f"paused_days={final_metrics.get('days_after_breaker', 0)}  "
            f"cooldowns={final_metrics.get('cooldown_trigger_count', 0)}"
        )
    if "dsr" in metrics:
        print(
            f"    risk-adjusted: sharpe={metrics.get('sharpe', 0):+.3f}  "
            f"ann.={metrics.get('annualized_sharpe', 0):+.2f}  "
            f"PSR={metrics.get('psr', 0):.3f}  "
            f"DSR({metrics.get('dsr_trials', DEFAULT_SELECTION_TRIALS)})={metrics.get('dsr', 0):.3f}"
        )
    print_breaker_risk_report(metrics)


def print_breaker_risk_report(metrics):
    if not metrics.get("breaker_triggered") or metrics.get("breaker_trigger_date") is None:
        return
    traded = metrics.get("traded")
    if traded is None or traded.empty:
        return

    breaker_year = pd.to_datetime(metrics["breaker_trigger_date"]).year
    year_trades = traded[
        (pd.to_datetime(traded["date"]) >= pd.Timestamp(f"{breaker_year}-01-01"))
        & (pd.to_datetime(traded["date"]) <= pd.Timestamp(f"{breaker_year}-12-31"))
    ].copy()
    if year_trades.empty:
        return

    by_ticker = (
        year_trades.groupby("ticker")["strategy_return"]
        .agg(["count", "mean", "sum"])
        .sort_values("sum")
        .head(6)
    )
    rows = [
        [ticker, int(row["count"]), f"{row['mean'] * 100:+.2f}%", f"{row['sum'] * 100:+.2f}%"]
        for ticker, row in by_ticker.iterrows()
    ]
    print(f"    breaker risk report: {breaker_year} worst contributors")
    print(tabulate(rows, headers=["Ticker", "Trades", "Avg", "Total"], tablefmt="simple"))


def print_by_ticker(traded):
    if traded.empty:
        return
    section("OUT-OF-SAMPLE RESULTS BY TICKER")
    rows = []
    for ticker, group in traded.groupby("ticker"):
        acc = group["correct"].mean() * 100
        avg = group["strategy_return"].mean() * 100
        size = group["position_size"].mean() if "position_size" in group else 1
        rows.append([ticker, len(group), f"{acc:.1f}%", f"{avg:+.2f}%", f"{size:.2f}x"])
    print(tabulate(rows, headers=["Ticker", "Trades", "Accuracy", "Avg Return", "Avg Size"], tablefmt="simple"))


def print_feature_importance(model_name, model, data):
    section("FEATURE IMPORTANCE")
    try:
        cutoff = int(len(data) * 0.75)
        X_train = data[ACTIVE_FEATURE_COLUMNS].iloc[:cutoff]
        y_train = data["target"].iloc[:cutoff]
        X_test = data[ACTIVE_FEATURE_COLUMNS].iloc[cutoff:]
        y_test = data["target"].iloc[cutoff:]
        model.fit(X_train, y_train)
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=8,
            random_state=42,
            scoring="accuracy",
            n_jobs=-1,
        )
        rows = sorted(zip(ACTIVE_FEATURE_COLUMNS, result.importances_mean), key=lambda x: x[1], reverse=True)[:12]
        print(tabulate([[name, f"{score:.4f}"] for name, score in rows], headers=["Feature", "Importance"], tablefmt="simple"))
    except Exception as exc:
        print(f"  {Fore.YELLOW}Feature importance unavailable for {model_name}: {exc}{Style.RESET_ALL}")


def plot_equity(portfolio_returns):
    if not HAS_PLOT or portfolio_returns is None or len(portfolio_returns) == 0:
        return
    portfolio_returns = portfolio_returns.sort_index()
    equity = (1 + portfolio_returns).cumprod()
    plt.figure(figsize=(12, 5), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    ax.plot(portfolio_returns.index, equity, color="#58a6ff", linewidth=2)
    ax.set_title("Walk-Forward Strategy Equity", color="#f0f6fc")
    ax.set_ylabel("Growth of $1", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"  {Fore.GREEN}Equity chart saved -> {CHART_FILE}{Style.RESET_ALL}")


def trade_quality(prob_up, signal, whale_score, risk_on_score, row=None):
    if signal == 0:
        return "WAIT"
    if signal == 1:
        confidence_edge = max(0, (prob_up - 0.55) / 0.15)
    else:
        confidence_edge = max(0, (0.45 - prob_up) / 0.15)
    confidence_edge = min(confidence_edge, 1)

    whale_alignment = whale_score if signal == 1 else -whale_score
    regime_alignment = risk_on_score if signal == 1 else 1 - risk_on_score

    row = row if row is not None else {}
    setup_bonus = 0
    if row.get("setup_candidate", False):
        setup_bonus += 0.10
    if row.get("pocket_pivot", 0):
        setup_bonus += 0.10
    if signal == 1 and row.get("quiet_accumulation", 0):
        setup_bonus += 0.05
    if signal == -1 and row.get("quiet_distribution", 0):
        setup_bonus += 0.05
    if signal == 1 and row.get("vwap_reclaim", 0):
        setup_bonus += 0.05
    if signal == -1 and row.get("vwap_rejection", 0):
        setup_bonus += 0.05
    if signal == 1 and row.get("breakout_volume_20", 0):
        setup_bonus += 0.05
    if signal == -1 and row.get("breakdown_volume_20", 0):
        setup_bonus += 0.05
    setup_bonus = min(setup_bonus, 0.20)

    score = (
        confidence_edge * 35
        + max(0, whale_alignment) * 35
        + regime_alignment * 20
        + setup_bonus * 100
    )
    if score >= 75:
        return "A"
    if score >= 55:
        return "B"
    if score >= 40:
        return "C"
    return "LOW"


def actionable_signal(model_signal, quality, setup_candidate=True, row=None):
    if model_signal == 0 or not setup_candidate:
        return 0
    if quality in ACTIONABLE_QUALITY_LEVELS:
        return model_signal

    row = row if row is not None else {}
    breadth_score = safe_float(row.get("market_breadth_score"), 0.5)
    whale_score = safe_float(row.get("whale_footprint_score", row.get("whale_score")), 0)
    relative_strength = safe_float(row.get("relative_strength_20d"), 0)

    relaxed_long = (
        model_signal == 1
        and quality == "C"
        and (
            breadth_score >= 0.50
            or whale_score >= 0.20
            or relative_strength >= 0.05
        )
    )
    relaxed_short = (
        model_signal == -1
        and quality == "C"
        and (
            breadth_score <= 0.45
            or whale_score <= -0.20
            or relative_strength <= -0.05
        )
    )
    if relaxed_long or relaxed_short:
        return model_signal
    return 0


def near_up_action(prob_up, up_threshold, quality, setup_candidate):
    return (
        setup_candidate
        and quality in ("C", "LOW")
        and prob_up >= up_threshold - NEAR_UP_BUFFER
    )


def live_trade_rank(record):
    signal = record.get("signal_raw", 0)
    prob_up = record.get("prob_up", 0.5)
    whale = record.get("whale_score", 0)
    rs = record.get("relative_strength_20d", 0)
    prob_edge = prob_up - 0.5 if signal == 1 else 0.5 - prob_up
    whale_alignment = whale if signal == 1 else -whale
    rs_alignment = rs if signal == 1 else -rs
    setup_score = (
        record.get("pocket_pivot", 0) * 15
        + record.get("breakout_volume_20", 0) * 10
        + record.get("quiet_accumulation", 0) * 8
        + record.get("vwap_reclaim", 0) * 8
        + int(bool(record.get("setup_candidate", False))) * 5
    )
    return (
        max(0, prob_edge) * 100
        + max(0, whale_alignment) * 30
        + max(0, rs_alignment) * 40
        + record.get("position_size", 0) * 10
        + setup_score
    )


def apply_live_trade_limiter(
    records,
    max_daily_trades=DEFAULT_MAX_DAILY_TRADES,
    max_daily_sector_trades=DEFAULT_MAX_DAILY_SECTOR_TRADES,
):
    actionable = [record for record in records if record.get("signal") in ("UP", "DOWN")]
    if len(actionable) <= max_daily_trades:
        return records

    selected_ids = set()
    by_sector = {}
    for record in actionable:
        by_sector.setdefault(sector_for_ticker(record["ticker"]), []).append(record)

    sector_candidates = []
    for sector_records in by_sector.values():
        sector_candidates.extend(
            sorted(sector_records, key=live_trade_rank, reverse=True)[:max_daily_sector_trades]
        )
    day_max_trades = max_daily_trades
    risk_scores = [record.get("risk_on_score", 0.5) for record in actionable]
    if risk_scores and np.mean(risk_scores) < 0.45:
        day_max_trades = min(day_max_trades, 1)
    selected = sorted(sector_candidates, key=live_trade_rank, reverse=True)[:day_max_trades]
    selected_ids = {record["ticker"] for record in selected}

    for record in records:
        if record.get("signal") in ("UP", "DOWN") and record["ticker"] not in selected_ids:
            record["limited_out"] = True
            record["signal"] = "HOLD"
            record["signal_raw"] = 0
            record["position_size"] = 0
            record["color"] = Fore.YELLOW
    return records


def prepare_live_features(ticker, period="2y"):
    df = fetch(ticker, period=period)
    if len(df) < 260:
        return df, pd.DataFrame()
    feats = make_features(df)
    bench = make_benchmark_features(period)
    if not bench.empty:
        feats = feats.join(bench, how="left")
    fundamental = fetch_fundamental_snapshot(ticker)
    for col, value in fundamental.items():
        feats[col] = value
    kronos_feats = kronos_forecast_features(ticker, df)
    for col, value in kronos_feats.items():
        feats[col] = value
    for col in ["peer_ret_5d", "peer_ret_20d"]:
        feats[col] = 0
    feats["relative_strength_20d"] = feats["ret_20d"]
    feats = add_whale_score(feats)
    feats = add_regime_labels(feats)
    feats["setup_candidate"] = setup_candidate_mask(feats).astype(int)
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0
    feats[FEATURE_COLUMNS] = feats[FEATURE_COLUMNS].ffill().fillna(0)
    return df, feats.dropna(subset=ACTIVE_FEATURE_COLUMNS)


import sys


def register_model_namespace():
    """Make pickled model classes resolvable no matter which module name the
    .joblib file was saved under.

    Older model files reference the custom estimators both as ``main.X`` (from
    runs where the code lived in a module named ``main``) and as
    ``__main__.X`` (from script runs). Joblib pickles by module name, so we
    alias ``main`` to this module and mirror the estimator classes onto the
    current ``__main__`` module.
    """
    mod = sys.modules[__name__]
    sys.modules.setdefault("main", mod)

    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        import __main__
        main_mod = __main__
    for cls_name in ("AverageProbabilityEnsemble", "AttentionNetClassifier", "_AttentionNet"):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            setattr(main_mod, cls_name, cls)


def load_trained_payload():
    if not MODEL_FILE.exists():
        print("No model")
        return None

    register_model_namespace()

    payload = joblib.load(MODEL_FILE)
    return payload

def select_payload_model(payload, latest_row):
    regime = market_regime_label(latest_row)
    regime_models = payload.get("regime_models") or {}
    model = regime_models.get(regime)
    if model is not None:
        return model, regime, "specialist"
    return payload["model"], regime, "general"


def safe_float(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def score_fundamentals(info):
    if not info:
        return 0, []

    score = 0
    notes = []
    revenue_growth = safe_float(info.get("revenueGrowth"), None)
    earnings_growth = safe_float(info.get("earningsGrowth"), None)
    profit_margins = safe_float(info.get("profitMargins"), None)
    debt_to_equity = safe_float(info.get("debtToEquity"), None)
    forward_pe = safe_float(info.get("forwardPE"), None)
    recommendation = str(info.get("recommendationKey", "")).lower()

    if revenue_growth is not None:
        if revenue_growth > 0.15:
            score += 1
            notes.append("strong revenue growth")
        elif revenue_growth < -0.05:
            score -= 1
            notes.append("revenue contraction")

    if earnings_growth is not None:
        if earnings_growth > 0.10:
            score += 1
            notes.append("earnings growth positive")
        elif earnings_growth < -0.10:
            score -= 1
            notes.append("earnings growth weak")

    if profit_margins is not None:
        if profit_margins > 0.15:
            score += 1
            notes.append("healthy margins")
        elif profit_margins < 0:
            score -= 1
            notes.append("negative margins")

    if debt_to_equity is not None:
        if debt_to_equity > 200:
            score -= 1
            notes.append("high debt/equity")
        elif 0 < debt_to_equity < 60:
            score += 0.5
            notes.append("manageable debt/equity")

    if forward_pe is not None:
        if forward_pe > 80:
            score -= 0.5
            notes.append("expensive forward PE")
        elif 0 < forward_pe < 25:
            score += 0.5
            notes.append("reasonable forward PE")

    if recommendation in ("buy", "strong_buy"):
        score += 0.5
        notes.append("analyst tone positive")
    elif recommendation in ("sell", "strong_sell"):
        score -= 0.5
        notes.append("analyst tone negative")

    return max(-3, min(3, score)), notes[:4]


def classify_headline(title):
    text = title.lower()
    score = 0
    tags = []

    for tag, weight, phrases in NEWS_CATEGORY_RULES:
        if any(phrase in text for phrase in phrases):
            score += weight
            tags.append(tag)

    if not tags:
        pos = sum(1 for word in POSITIVE_NEWS_WORDS if word in text)
        neg = sum(1 for word in NEGATIVE_NEWS_WORDS if word in text)
        if pos or neg:
            score += (pos - neg) * 0.35
            tags.append("headline_tone")

    price_up = any(phrase in text for phrase in PRICE_ACTION_WORDS["up"])
    price_down = any(phrase in text for phrase in PRICE_ACTION_WORDS["down"])
    if price_down and score > 0:
        score -= 0.7
        tags.append("good_news_sold")
    elif price_up and score < 0:
        score += 0.4
        tags.append("bad_news_bought")

    return score, tags


def score_news(news_items):
    if not news_items:
        return 0, []
    titles = []
    for item in news_items[:8]:
        title = item.get("title") if isinstance(item, dict) else ""
        if title:
            titles.append(title)
    finbert_results = finbert_score_titles(titles)
    score = 0
    notes = []
    for i, title in enumerate(titles):
        headline_score, tags = classify_headline(title)
        if i < len(finbert_results):
            finbert_score, finbert_tag, finbert_confidence = finbert_results[i]
            combined_score = headline_score * 0.45 + finbert_score * 1.25
            tags = [finbert_tag] + tags
            if finbert_tag != "finbert_neutral":
                tags.append(f"conf_{finbert_confidence:.2f}")
        else:
            combined_score = headline_score
        score += combined_score
        if tags:
            tag_text = ",".join(tags[:2])
            notes.append(f"{tag_text}: {title[:90]}")
    return max(-3, min(3, score)), notes[:3]


def earnings_event_risk(ticker_obj):
    try:
        calendar = ticker_obj.calendar
    except Exception:
        return 0, "unknown"

    dates = []
    try:
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            for value in calendar.to_numpy().flatten():
                if pd.notna(value):
                    dates.append(pd.Timestamp(value))
        elif isinstance(calendar, dict):
            for value in calendar.values():
                if isinstance(value, (list, tuple)):
                    dates.extend(pd.Timestamp(v) for v in value if pd.notna(v))
                elif pd.notna(value):
                    dates.append(pd.Timestamp(value))
    except Exception:
        return 0, "unknown"

    if not dates:
        return 0, "none"

    now = pd.Timestamp.today().normalize()
    future_dates = sorted(date.normalize() for date in dates if date.normalize() >= now)
    if not future_dates:
        return 0, "none"

    days = int((future_dates[0] - now).days)
    if days <= 3:
        return -2, f"earnings/event in {days}d"
    if days <= 10:
        return -1, f"earnings/event in {days}d"
    return 0, f"event in {days}d"


def finnhub_metric_info(ticker):
    data = finnhub_get("/stock/metric", {"symbol": ticker, "metric": "all"})
    if not data or not isinstance(data, dict):
        return {}
    metric = data.get("metric", {}) or {}
    revenue_growth = safe_float(metric.get("revenueGrowthTTMYoy"), None)
    earnings_growth = safe_float(metric.get("epsGrowthTTMYoy"), None)
    profit_margins = safe_float(metric.get("netProfitMarginTTM"), None)
    if revenue_growth is not None and abs(revenue_growth) > 1:
        revenue_growth /= 100
    if earnings_growth is not None and abs(earnings_growth) > 1:
        earnings_growth /= 100
    if profit_margins is not None and abs(profit_margins) > 1:
        profit_margins /= 100
    return {
        "revenueGrowth": revenue_growth,
        "earningsGrowth": earnings_growth,
        "profitMargins": profit_margins,
        "debtToEquity": safe_float(metric.get("totalDebt/totalEquityAnnual"), None),
        "forwardPE": safe_float(metric.get("forwardPE"), None),
        "recommendationKey": "",
    }


FUNDAMENTAL_SNAPSHOT_CACHE = {}


def fetch_fundamental_snapshot(ticker):
    if ticker in FUNDAMENTAL_SNAPSHOT_CACHE:
        return FUNDAMENTAL_SNAPSHOT_CACHE[ticker]

    default = {
        "fund_rev_growth": 0.0,
        "fund_eps_growth": 0.0,
        "fund_profit_margin": 0.0,
        "fund_debt_equity": 0.0,
        "fund_forward_pe": 0.0,
        "fund_rec_score": 0.5,
    }
    info = {}
    if finnhub_api_key():
        info = finnhub_metric_info(ticker)
    if not info:
        try:
            info = getattr(yf.Ticker(ticker), "info", {}) or {}
        except Exception:
            info = {}

    rev_growth = safe_float(info.get("revenueGrowth"), None)
    eps_growth = safe_float(info.get("earningsGrowth"), None)
    profit_margin = safe_float(info.get("profitMargins"), None)
    debt_equity = safe_float(info.get("debtToEquity"), None)
    forward_pe = safe_float(info.get("forwardPE"), None)
    recommendation = str(info.get("recommendationKey", "")).lower()

    rec_map = {
        "strong_buy": 1.0,
        "buy": 1.0,
        "hold": 0.5,
        "neutral": 0.5,
        "underperform": 0.25,
        "sell": 0.0,
        "strong_sell": 0.0,
    }
    snapshot = {
        "fund_rev_growth": float(np.clip(rev_growth, -1, 2)) if rev_growth is not None else 0.0,
        "fund_eps_growth": float(np.clip(eps_growth, -2, 3)) if eps_growth is not None else 0.0,
        "fund_profit_margin": float(np.clip(profit_margin, -1, 1)) if profit_margin is not None else 0.0,
        "fund_debt_equity": float(np.clip(debt_equity / 100, 0, 5)) if debt_equity is not None else 0.0,
        "fund_forward_pe": float(np.clip(forward_pe / 50, 0, 3)) if forward_pe is not None else 0.0,
        "fund_rec_score": float(rec_map.get(recommendation, 0.5)),
    }
    FUNDAMENTAL_SNAPSHOT_CACHE[ticker] = snapshot
    return snapshot


def finnhub_company_news(ticker):
    today = dt.date.today()
    start = today - dt.timedelta(days=14)
    data = finnhub_get(
        "/company-news",
        {"symbol": ticker, "from": start.isoformat(), "to": today.isoformat()},
        ttl_seconds=60 * 20,
    )
    if not isinstance(data, list):
        return []
    return [{"title": item.get("headline", ""), "source": item.get("source", "")} for item in data[:12]]


def finnhub_earnings_event_risk(ticker):
    today = dt.date.today()
    end = today + dt.timedelta(days=30)
    data = finnhub_get(
        "/calendar/earnings",
        {"symbol": ticker, "from": today.isoformat(), "to": end.isoformat()},
        ttl_seconds=60 * 60 * 6,
    )
    if not isinstance(data, dict):
        return None
    earnings = data.get("earningsCalendar", []) or []
    future = []
    for item in earnings:
        date_text = item.get("date")
        if not date_text:
            continue
        try:
            future.append(pd.Timestamp(date_text).normalize())
        except Exception:
            continue
    if not future:
        return 0, "none"
    next_date = sorted(future)[0]
    days = int((next_date - pd.Timestamp.today().normalize()).days)
    if days <= 3:
        return -2, f"earnings in {days}d"
    if days <= 10:
        return -1, f"earnings in {days}d"
    return 0, f"earnings in {days}d"


def market_intelligence(ticker):
    provider_notes = []
    info = {}
    news_items = []
    event_result = None

    if finnhub_api_key():
        info = finnhub_metric_info(ticker)
        news_items = finnhub_company_news(ticker)
        event_result = finnhub_earnings_event_risk(ticker)
        provider_notes.append("finnhub")

    try:
        ticker_obj = yf.Ticker(ticker)
        if not info:
            info = getattr(ticker_obj, "info", {}) or {}
            provider_notes.append("yfinance")
    except Exception as exc:
        ticker_obj = None
        if not info:
            return {
                "fundamental_score": 0,
                "news_score": 0,
                "event_score": 0,
                "intelligence_score": 0,
                "fundamental_notes": [f"fundamentals unavailable: {exc}"],
                "news_notes": [],
                "event_note": "unknown",
                "intelligence_provider": ",".join(provider_notes) or "none",
            }

    fundamental_score, fundamental_notes = score_fundamentals(info)
    if not news_items and ticker_obj is not None:
        try:
            news_items = ticker_obj.news or []
        except Exception:
            news_items = []
    news_score, news_notes = score_news(news_items)
    if event_result is not None:
        event_score, event_note = event_result
    elif ticker_obj is not None:
        event_score, event_note = earnings_event_risk(ticker_obj)
    else:
        event_score, event_note = 0, "unknown"
    intelligence_score = fundamental_score * 0.45 + news_score * 0.30 + event_score * 0.25

    return {
        "fundamental_score": fundamental_score,
        "news_score": news_score,
        "event_score": event_score,
        "intelligence_score": max(-3, min(3, intelligence_score)),
        "fundamental_notes": fundamental_notes,
        "news_notes": news_notes,
        "event_note": event_note,
        "intelligence_provider": ",".join(provider_notes) or "yfinance",
    }


def regime_exposure_scale(regime, risk_on=0.5, custom=None):
    """Beta / regime de-risking: scale gross exposure by market regime.

    Bearish and high-volatility regimes cut gross exposure hard; trending
    regimes allow full exposure. A continuous risk_on term nudges the scale so
    a borderline regime still moves exposure smoothly rather than flipping a switch.
    """
    base = {
        "bear": 0.40,
        "high_volatility": 0.60,
        "neutral": 0.80,
        "mean_reversion": 0.90,
        "trend": 1.00,
    }
    scale = base.get(regime, 0.80)
    if custom and regime in custom:
        scale = float(custom[regime])
    ro = safe_float(risk_on, 0.5)
    blended = scale * (0.5 + ro)
    return float(min(1.0, max(0.0, blended)))


def _greedy_select(candidates, cap, corr_penalty, side):
    """Pick the top-ranked names by live_trade_rank, applying a simple
    momentum-crowding (correlation) penalty. Sector exposure is enforced later
    on the final weights via _enforce_sector_cap."""
    if not candidates or cap <= 0:
        return []
    ranked = sorted(candidates, key=live_trade_rank, reverse=True)
    chosen = []
    chosen_rs = []
    for r in ranked:
        if len(chosen) >= cap:
            break
        rs = safe_float(r.get("relative_strength_20d"), 0)
        if corr_penalty > 0 and len(chosen_rs) >= 2 and min(abs(rs - o) for o in chosen_rs) < 0.01:
            # Momentum-factor crowding: too similar to names already held.
            continue
        chosen.append(r)
        chosen_rs.append(rs)
    return chosen


def _enforce_sector_cap(book, max_sector, gross_budget, side_budget):
    """Drop smallest-weight names until no sector exceeds max_sector of the
    gross budget, then renormalize the survivors to the side budget."""
    if not book or side_budget <= 0:
        return book
    threshold = max_sector * gross_budget
    ordered = sorted(book, key=lambda c: c["weight"], reverse=True)
    kept = []
    sector_used = {}
    for c in ordered:
        sec = sector_for_ticker(c["ticker"])
        if sector_used.get(sec, 0.0) + c["weight"] > threshold + 1e-9:
            continue
        kept.append(c)
        sector_used[sec] = sector_used.get(sec, 0.0) + c["weight"]
    total = sum(c["weight"] for c in kept)
    if total > 0:
        for c in kept:
            c["weight"] = c["weight"] / total * side_budget
    return kept


def _assign_weights(selected, budget, min_name, side):
    if not selected or budget <= 0:
        return []
    total = sum((float(r.get("position_size", 0)) or 0) for r in selected)
    out = []
    for r in selected:
        if total > 0:
            w = (float(r.get("position_size", 0)) or 0) / total * budget
        else:
            w = budget / len(selected)
        w = max(w, min(min_name, budget / len(selected)))
        out.append({
            "ticker": r["ticker"],
            "side": side,
            "signal": r.get("signal"),
            "signal_raw": r.get("signal_raw", 1 if side == "LONG" else -1),
            "quality": r.get("quality"),
            "prob_up": r.get("prob_up"),
            "position_size": r.get("position_size"),
            "stop_loss": r.get("stop_loss"),
            "take_profit": r.get("take_profit"),
            "price": r.get("price"),
            "weight": w,
        })
    s = sum(o["weight"] for o in out)
    if s > 0:
        for o in out:
            o["weight"] = o["weight"] / s * budget
    return out


def select_portfolio(records, cfg):
    """Portfolio-aware long/short selection across the universe.

    Returns a list of dicts with target weights and side, subject to per-side
    name caps, sector exposure caps, a momentum-crowding correlation penalty,
    and a regime-driven gross-exposure scaler.
    """
    if not records:
        return []
    allow_short = bool(cfg.get("allow_short", False))

    longs = [r for r in records if r.get("signal") == "UP"]
    shorts = [r for r in records if r.get("signal") == "DOWN"] if allow_short else []

    max_long = int(cfg.get("max_long_positions", 12))
    max_short = int(cfg.get("max_short_positions", 8))
    max_sector = float(cfg.get("max_sector_exposure", 0.20))
    min_name = float(cfg.get("min_name_weight", 0.02))
    corr_penalty = float(cfg.get("correlation_penalty", 0.5))

    target_gross = float(cfg.get("target_gross_exposure", 1.0))
    long_budget = target_gross * float(cfg.get("long_weight", 0.65))
    short_budget = target_gross * float(cfg.get("short_weight", 0.35))

    risk_on = np.mean([safe_float(r.get("risk_on_score"), 0.5) for r in records]) if records else 0.5
    regimes = [r.get("active_regime") for r in records if r.get("active_regime")]
    regime = max(set(regimes), key=regimes.count) if regimes else "neutral"
    scale = regime_exposure_scale(regime, risk_on, cfg.get("regime_exposure_scale"))

    longs_sel = _greedy_select(longs, max_long, corr_penalty, "LONG")
    shorts_sel = _greedy_select(shorts, max_short, corr_penalty, "SHORT")

    lbook = _assign_weights(longs_sel, long_budget * scale, min_name, "LONG")
    sbook = _assign_weights(shorts_sel, short_budget * scale, min_name, "SHORT")
    gross_budget = target_gross * scale
    lbook = _enforce_sector_cap(lbook, max_sector, gross_budget, long_budget * scale)
    sbook = _enforce_sector_cap(sbook, max_sector, gross_budget, short_budget * scale)

    chosen = lbook + sbook
    for c in chosen:
        c["regime"] = regime
        c["regime_scale"] = round(scale, 3)
    return chosen


def live_prediction_records(tickers):
    payload = load_trained_payload()
    if payload is None:
        return [], None

    up_threshold = payload["prob_up"]
    down_threshold = payload["prob_down"]
    stop_loss = payload.get("stop_loss", DEFAULT_STOP_LOSS)
    take_profit = payload.get("take_profit", DEFAULT_TAKE_PROFIT)
    max_position_size = payload.get("max_position_size", DEFAULT_MAX_POSITION_SIZE)
    training_mode = payload.get("training_mode", "all")
    records = []

    for ticker in tickers:
        df, feats = prepare_live_features(ticker)
        if feats.empty:
            continue
        latest = feats[payload.get("features", ACTIVE_FEATURE_COLUMNS)].iloc[[-1]]
        latest_row = feats.iloc[-1]
        alloc = allocated_live_probability(payload, latest_row, latest)
        raw_prob_up = alloc["raw_prob_up"]
        prob_up = alloc["prob_up"]
        active_regime = alloc["regime"]
        model_route = alloc["model_route"]
        model_signal = signal_from_probability(prob_up, up_threshold, down_threshold)
        model_direction = "UP" if model_signal == 1 else ("DOWN" if model_signal == -1 else "HOLD")
        setup_candidate = bool(latest_row.get("setup_candidate", 1))
        quality = trade_quality(
            prob_up,
            model_signal,
            float(latest_row.get("whale_footprint_score", 0)),
            float(latest_row.get("risk_on_score", 0.5)),
            latest_row,
        )
        signal = actionable_signal(
            model_signal,
            quality,
            setup_candidate if training_mode == "setups" else True,
            latest_row,
        )
        near_up = signal == 0 and near_up_action(
            prob_up,
            up_threshold,
            quality,
            setup_candidate if training_mode == "setups" else True,
        )
        direction = "UP" if signal == 1 else ("DOWN" if signal == -1 else ("NEAR" if near_up else "HOLD"))
        color = Fore.GREEN if signal == 1 else (Fore.RED if signal == -1 else (Fore.LIGHTGREEN_EX if near_up else Fore.YELLOW))
        size = position_size(prob_up, signal, up_threshold, down_threshold, max_position_size)
        whale_score = float(latest_row.get("whale_footprint_score", 0))
        risk_on_score = float(latest_row.get("risk_on_score", 0.5))
        trend_score = float(latest_row.get("trend_score", 0.5))
        volatility_score = float(latest_row.get("volatility_score", 0.5))
        market_breadth_score = float(latest_row.get("market_breadth_score", 0.5))
        credit_stress_score = float(latest_row.get("credit_stress_score", 0.5))
        rsi_value = float(latest_row.get("rsi_14", 50))
        relative_strength = float(latest_row.get("relative_strength_20d", 0))
        sma20_dist = float(latest_row.get("sma_20_dist", 0))
        sma50_dist = float(latest_row.get("sma_50_dist", 0))
        intel = market_intelligence(ticker)
        records.append(
            {
                "date": df.index[-1].date().isoformat(),
                "ticker": ticker,
                "price": float(df["Close"].iloc[-1]),
                "signal": direction,
                "signal_raw": signal,
                "near_up": near_up,
                "model_signal": model_direction,
                "model_signal_raw": model_signal,
                "raw_prob_up": raw_prob_up,
                "prob_up": prob_up,
                "quality": quality,
                "setup_candidate": setup_candidate,
                "pocket_pivot": int(bool(latest_row.get("pocket_pivot", 0))),
                "breakout_volume_20": int(bool(latest_row.get("breakout_volume_20", 0))),
                "quiet_accumulation": int(bool(latest_row.get("quiet_accumulation", 0))),
                "vwap_reclaim": int(bool(latest_row.get("vwap_reclaim", 0))),
                "training_mode": training_mode,
                "active_regime": active_regime,
                "model_route": model_route,
                "model_probs": alloc.get("model_probs", {}),
                "capital_shares": alloc.get("capital_shares", {}),
                "agreement": alloc.get("agreement", 1.0),
                "position_size": size,
                "whale_score": whale_score,
                "risk_on_score": risk_on_score,
                "trend_score": trend_score,
                "volatility_score": volatility_score,
                "market_breadth_score": market_breadth_score,
                "credit_stress_score": credit_stress_score,
                "relative_strength_20d": relative_strength,
                "rsi": rsi_value,
                "sma20_dist": sma20_dist,
                "sma50_dist": sma50_dist,
                "fundamental_score": intel["fundamental_score"],
                "news_score": intel["news_score"],
                "event_score": intel["event_score"],
                "intelligence_score": intel["intelligence_score"],
                "fundamental_notes": intel["fundamental_notes"],
                "news_notes": intel["news_notes"],
                "event_note": intel["event_note"],
                "intelligence_provider": intel["intelligence_provider"],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "horizon_days": int(payload["horizon_days"]),
                "model_trained_at": payload.get("trained_at", ""),
                "color": color,
            }
        )

    records = apply_live_trade_limiter(records)
    return records, payload


def is_top_setup(record):
    signal = record["signal"]
    if signal not in ("UP", "DOWN") or record["quality"] not in TOP_QUALITY_LEVELS:
        return False
    whale_ok = record["whale_score"] > 0 if signal == "UP" else record["whale_score"] < 0
    regime_ok = record["risk_on_score"] >= 0.50 if signal == "UP" else record["risk_on_score"] <= 0.50
    return whale_ok and regime_ok and record["position_size"] >= 0.5


def explain_record(record):
    notes = []
    whale = record["whale_score"]
    rsi_value = record["rsi"]
    risk_on = record["risk_on_score"]
    prob_up = record["prob_up"]
    sma20_dist = record["sma20_dist"]
    signal = record["signal"]
    intel_score = record.get("intelligence_score", 0)
    news_score = record.get("news_score", 0)
    fundamental_score = record.get("fundamental_score", 0)
    event_note = record.get("event_note", "unknown")

    if whale >= 0.40:
        notes.append("strong bullish footprint")
    elif whale >= 0.20:
        notes.append("bullish footprint")
    elif whale <= -0.40:
        notes.append("strong bearish footprint")
    elif whale <= -0.20:
        notes.append("bearish footprint")
    else:
        notes.append("neutral whale footprint")

    if rsi_value < 35 and whale > 0.20:
        notes.append("RSI low + bullish whale = possible underbought accumulation")
    elif rsi_value > 70 and whale > 0.20:
        notes.append("RSI high + bullish whale = momentum, but chase risk")
    elif rsi_value > 65 and whale < -0.20:
        notes.append("RSI high + bearish whale = possible distribution after strength")
    elif rsi_value < 35 and whale < -0.20:
        notes.append("RSI low + bearish whale = weak tape, avoid catching the fall")
    elif 45 <= rsi_value <= 55:
        notes.append("RSI neutral = no strong momentum clue")

    if sma20_dist > 0.08:
        notes.append("price extended above 20-day average")
    elif sma20_dist < -0.08:
        notes.append("price stretched below 20-day average")

    if record.get("limited_out"):
        notes.append("passed signal gate but skipped by daily trade/sector limiter")

    if risk_on >= 0.75:
        notes.append("market regime supportive")
    elif risk_on <= 0.25:
        notes.append("market regime defensive")

    if signal == "HOLD":
        if prob_up >= 0.58:
            notes.append("model leans bullish but threshold not reached")
        elif prob_up <= 0.42:
            notes.append("model leans bearish but threshold not reached")
        else:
            notes.append("model sees no clear edge")

    if intel_score >= 1:
        notes.append("market intelligence supportive")
    elif intel_score <= -1:
        notes.append("market intelligence cautionary")

    if news_score >= 1:
        notes.append("recent headlines lean positive")
    elif news_score <= -1:
        notes.append("recent headlines lean negative/risky")

    if fundamental_score >= 1.5:
        notes.append("fundamentals supportive")
    elif fundamental_score <= -1.5:
        notes.append("fundamentals weak/cautionary")

    if event_note not in ("none", "unknown"):
        notes.append(event_note)

    return "; ".join(notes)


def print_record_explanations(records, limit=8):
    if not records:
        return
    ranked = sorted(
        records,
        key=lambda row: (
            row["signal"] in ("UP", "DOWN"),
            abs(row["whale_score"]),
            abs(row["prob_up"] - 0.5),
        ),
        reverse=True,
    )[:limit]
    section("SIGNAL INTERPRETATION")
    for record in ranked:
        print(f"  {record['ticker']:<6} {explain_record(record)}")
        details = []
        if record.get("fundamental_notes"):
            details.append("fund: " + ", ".join(record["fundamental_notes"][:2]))
        if record.get("news_notes"):
            details.append("news: " + " | ".join(record["news_notes"][:1]))
        if record.get("intelligence_provider"):
            details.append("provider: " + record["intelligence_provider"])
        if details:
            print(f"         {'; '.join(details)}")


def intelligence_report_score(record):
    direction_bias = (record["prob_up"] - 0.5) * 100
    whale = record["whale_score"] * 18
    regime = (record["risk_on_score"] - 0.5) * 16
    intel = record.get("intelligence_score", 0) * 8
    rsi = record["rsi"]
    timing = 0
    if 40 <= rsi <= 65:
        timing += 4
    elif rsi > 75:
        timing -= 5
    elif rsi < 30:
        timing -= 2
    extension = abs(record["sma20_dist"])
    if extension > 0.10:
        timing -= 4
    conflict_penalty = 0
    if record_conflicts(record):
        conflict_penalty = 10
    return direction_bias + whale + regime + intel + timing - conflict_penalty


def record_conflicts(record):
    signal = record.get("signal", "HOLD")
    whale = record.get("whale_score", 0)
    rsi_value = record.get("rsi", 50)
    intel = record.get("intelligence_score", 0)
    if signal == "UP":
        return (whale <= -0.20 and rsi_value < 40) or intel <= -1
    if signal == "DOWN":
        return (whale >= 0.20 and rsi_value > 60) or intel >= 1
    return (whale <= -0.20 and rsi_value < 40) or (whale >= 0.20 and rsi_value > 70)


def intelligence_bias_label(record):
    score = intelligence_report_score(record)
    conflict = record_conflicts(record)
    if score >= 8:
        return "Caution Bullish" if conflict else "Bullish"
    if score <= -8:
        return "Caution Bearish" if conflict else "Bearish"
    return "Conflicted" if conflict else "Neutral"


def bearish_pressure_score(record):
    score = 0
    score += max(0, 0.5 - record.get("prob_up", 0.5)) * 40
    score += max(0, -record.get("whale_score", 0)) * 35
    score += max(0, 0.5 - record.get("risk_on_score", 0.5)) * 20
    score += max(0, -record.get("intelligence_score", 0)) * 8
    rsi_value = record.get("rsi", 50)
    sma20_dist = record.get("sma20_dist", 0)
    if rsi_value > 65 and record.get("whale_score", 0) < -0.15:
        score += 10
    if rsi_value < 40 and record.get("whale_score", 0) < -0.15:
        score += 8
    if sma20_dist < -0.05:
        score += 6
    if record.get("signal") == "DOWN":
        score += 12
    return score


def bearish_pressure_label(record):
    score = bearish_pressure_score(record)
    if score >= 30:
        return "High bearish pressure"
    if score >= 18:
        return "Bearish watch"
    if score >= 10:
        return "Mild caution"
    return "Low"


def market_intelligence_report(tickers):
    records, _ = live_prediction_records(tickers)
    if not records:
        return

    ranked = sorted(records, key=intelligence_report_score, reverse=True)
    section("MARKET INTELLIGENCE REPORT")
    rows = []
    for record in ranked:
        score = intelligence_report_score(record)
        bias = intelligence_bias_label(record)
        event = record.get("event_note", "unknown")
        event_text = "" if event in ("none", "unknown") else event
        rows.append(
            [
                record["ticker"],
                bias,
                f"{score:+.1f}",
                record["signal"],
                f"{record['prob_up'] * 100:.1f}%",
                f"{record['whale_score']:+.2f}",
                f"{record['risk_on_score']:.2f}",
                f"{record['rsi']:.1f}",
                f"{record.get('intelligence_score', 0):+.1f}",
                event_text,
            ]
        )

    print(
        tabulate(
            rows,
            headers=["Ticker", "Bias", "Score", "Signal", "Prob Up", "Whale", "Risk", "RSI", "Intel", "Event"],
            tablefmt="simple",
        )
    )

    section("BEST BULLISH CASES")
    bullish = [record for record in ranked if intelligence_report_score(record) > 0][:5]
    if not bullish:
        print("  No clear bullish cases.")
    for record in bullish:
        print(f"  {record['ticker']:<6} {explain_record(record)}")

    section("BIGGEST CAUTION CASES")
    bearish = sorted(records, key=intelligence_report_score)[:5]
    for record in bearish:
        print(f"  {record['ticker']:<6} {explain_record(record)}")


def bearish_watchlist(tickers):
    records, _ = live_prediction_records(tickers)
    if not records:
        return
    ranked = sorted(records, key=bearish_pressure_score, reverse=True)
    section("BEARISH WATCHLIST")
    rows = []
    for record in ranked:
        score = bearish_pressure_score(record)
        if score < 8:
            continue
        rows.append(
            [
                record["ticker"],
                bearish_pressure_label(record),
                f"{score:.1f}",
                record["signal"],
                f"{record['prob_up'] * 100:.1f}%",
                f"{record['whale_score']:+.2f}",
                f"{record['risk_on_score']:.2f}",
                f"{record['rsi']:.1f}",
                f"{record.get('intelligence_score', 0):+.1f}",
                explain_record(record),
            ]
        )
    if not rows:
        print("  No meaningful bearish pressure right now.")
        return
    print(
        tabulate(
            rows[:12],
            headers=["Ticker", "Pressure", "Score", "Signal", "Prob Up", "Whale", "Risk", "RSI", "Intel", "Reason"],
            tablefmt="simple",
        )
    )


def print_live_records(records, title):
    section(title)
    rows = []
    for record in records:
        rows.append(
            [
                record["ticker"],
                f"${record['price']:,.2f}",
                record.get("model_signal", record["signal"]),
                f"{record['color']}{record['signal']}{Style.RESET_ALL}",
                f"{record.get('raw_prob_up', record['prob_up']) * 100:.1f}%",
                f"{record['prob_up'] * 100:.1f}%",
                f"{record.get('active_regime', 'unknown')}/{record.get('model_route', 'general')}",
                "Y" if record.get("setup_candidate", True) else "N",
                record["quality"],
                f"{record['position_size']:.1f}x",
                f"{record['rsi']:.1f}",
                f"{record['whale_score']:+.2f}",
                f"{record['risk_on_score']:.2f}",
                f"{record['intelligence_score']:+.1f}",
                f"{record['stop_loss']:.0%}/{record['take_profit']:.0%}",
                f"{record['horizon_days']}d",
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "Ticker",
                "Price",
                "Model",
                "Action",
                "Raw Prob",
                "Prob Up",
                "Regime",
                "Setup",
                "Quality",
                "Size",
                "RSI",
                "Whale",
                "Risk-On",
                "Intel",
                "Stop/TP",
                "Horizon",
            ],
            tablefmt="simple",
        )
    )
    print_regime_layers(records)
    print_model_bias_warning(records)
    print_record_explanations(records)


def print_model_bias_warning(records):
    if not records:
        return
    model_signals = {record.get("model_signal") for record in records}
    probs = [record.get("prob_up", 0.5) for record in records]
    actionable = [record for record in records if record.get("signal") in ("UP", "DOWN")]
    if len(model_signals) == 1 and len(records) >= 5 and np.std(probs) < 0.015:
        print(
            f"  {Fore.YELLOW}Note: model probabilities are tightly clustered. "
            f"This is mostly a broad-market bias, not ticker-specific conviction."
            f"{Style.RESET_ALL}"
        )
    if not actionable:
        print(
            f"  {Fore.YELLOW}No actionable A/B setups. Raw model bias is being held back by quality gates."
            f"{Style.RESET_ALL}"
        )


def print_regime_layers(records):
    if not records:
        return
    first = records[0]
    if not all(key in first for key in ("trend_score", "volatility_score", "market_breadth_score", "credit_stress_score")):
        return
    section("MARKET REGIME LAYERS")
    rows = [
        ["Trend", f"{first['trend_score']:.2f}", "SPY, QQQ, SMH, XLI, XLB, ITB above 50-day trend"],
        ["Volatility", f"{first['volatility_score']:.2f}", "VIX level, VIX 5-day change, VIX vs 20-day trend"],
        ["Breadth", f"{first['market_breadth_score']:.2f}", "RSP/SPY equal-weight breadth and IWM/SPY small-cap participation"],
        ["Credit", f"{first['credit_stress_score']:.2f}", "HYG/IEF high-yield credit appetite vs Treasury safety"],
        ["Composite", f"{first['risk_on_score']:.2f}", "35% trend, 25% volatility, 25% breadth, 15% credit"],
    ]
    print(tabulate(rows, headers=["Layer", "Score", "Meaning"], tablefmt="simple"))


def live_predict(tickers):
    records, _ = live_prediction_records(tickers)
    if records:
        log_live_predictions_to_db(records)
        print_live_records(records, "LIVE PREDICTIONS")


def top_setups_only(tickers):
    records, _ = live_prediction_records(tickers)
    if not records:
        return
    top_records = [record for record in records if is_top_setup(record)]
    if not top_records:
        section("TOP SETUPS ONLY")
        print("  No A/B quality setups with aligned whale and regime filters right now.")
        return
    log_live_predictions_to_db(top_records)
    print_live_records(top_records, "TOP SETUPS ONLY")


def save_paper_predictions(tickers, top_only=False):
    records, payload = live_prediction_records(tickers)
    if not records or payload is None:
        return
    if top_only:
        records = [record for record in records if is_top_setup(record)]
    if not records:
        print(f"  {Fore.YELLOW}No qualifying paper-trade signals to save.{Style.RESET_ALL}")
        return

    rows = []
    saved_at = dt.datetime.now().isoformat()
    for record in records:
        target_date = pd.Timestamp(record["date"]) + pd.Timedelta(days=record["horizon_days"])
        rows.append(
            {
                "saved_at": saved_at,
                "signal_date": record["date"],
                "target_date": target_date.date().isoformat(),
                "ticker": record["ticker"],
                "entry_price": record["price"],
                "signal": record["signal"],
                "prob_up": record["prob_up"],
                "quality": record["quality"],
                "position_size": record["position_size"],
                "whale_score": record["whale_score"],
                "risk_on_score": record["risk_on_score"],
                "rsi": record["rsi"],
                "fundamental_score": record["fundamental_score"],
                "news_score": record["news_score"],
                "event_score": record["event_score"],
                "intelligence_score": record["intelligence_score"],
                "event_note": record["event_note"],
                "intelligence_provider": record["intelligence_provider"],
                "stop_loss": record["stop_loss"],
                "take_profit": record["take_profit"],
                "horizon_days": record["horizon_days"],
                "model_trained_at": record["model_trained_at"],
                "checked_at": "",
                "exit_price": "",
                "actual_return": "",
                "correct": "",
                "status": "OPEN",
            }
        )

    for row in rows:
        log_paper_trade_to_db(row)

    new_data = pd.DataFrame(rows)
    if PAPER_TRADES_FILE.exists():
        old_data = pd.read_csv(PAPER_TRADES_FILE)
        new_data = pd.concat([old_data, new_data], ignore_index=True, sort=False)
    new_data.to_csv(PAPER_TRADES_FILE, index=False)
    print(f"  {Fore.GREEN}Saved {len(rows)} paper prediction(s) -> {PAPER_TRADES_FILE}{Style.RESET_ALL}")


def pattern_tags_from_row(row):
    tags = []
    whale = safe_float(row.get("whale_score"), 0)
    risk_on = safe_float(row.get("risk_on_score"), 0.5)
    rsi_value = safe_float(row.get("rsi"), 50)
    intel = safe_float(row.get("intelligence_score"), 0)
    news = safe_float(row.get("news_score"), 0)
    event = str(row.get("event_note", ""))

    if whale >= 0.40:
        tags.append("strong_bullish_whale")
    elif whale >= 0.20:
        tags.append("bullish_whale")
    elif whale <= -0.40:
        tags.append("strong_bearish_whale")
    elif whale <= -0.20:
        tags.append("bearish_whale")
    else:
        tags.append("neutral_whale")

    if risk_on >= 0.75:
        tags.append("risk_on")
    elif risk_on <= 0.25:
        tags.append("risk_off")

    if rsi_value < 35:
        tags.append("rsi_low")
    elif rsi_value > 70:
        tags.append("rsi_high")
    else:
        tags.append("rsi_normal")

    if intel >= 1:
        tags.append("intel_supportive")
    elif intel <= -1:
        tags.append("intel_cautionary")

    if news >= 1:
        tags.append("positive_news")
    elif news <= -1:
        tags.append("negative_news")

    if "earnings" in event and "none" not in event:
        tags.append("earnings_nearby")

    signal = str(row.get("signal", "")).lower()
    quality = str(row.get("quality", "")).lower()
    if signal:
        tags.append(f"signal_{signal}")
    if quality:
        tags.append(f"quality_{quality}")
    return tags


def append_pattern_memory(row):
    memory = load_json_file(PATTERN_MEMORY_FILE, {"patterns": []})
    tags = pattern_tags_from_row(row)
    record = {
        "closed_at": dt.datetime.now().isoformat(),
        "ticker": row.get("ticker"),
        "signal_date": row.get("signal_date"),
        "target_date": row.get("target_date"),
        "signal": row.get("signal"),
        "quality": row.get("quality"),
        "setup": "+".join(tags[:5]),
        "tags": tags,
        "prob_up": safe_float(row.get("prob_up"), 0),
        "whale_score": safe_float(row.get("whale_score"), 0),
        "risk_on_score": safe_float(row.get("risk_on_score"), 0.5),
        "rsi": safe_float(row.get("rsi"), 50),
        "news_score": safe_float(row.get("news_score"), 0),
        "intelligence_score": safe_float(row.get("intelligence_score"), 0),
        "event_note": row.get("event_note", ""),
        "result": safe_float(row.get("actual_return"), 0),
        "correct": str(row.get("correct")).lower() == "true",
    }
    key = (record["ticker"], record["signal_date"], record["target_date"], record["signal"])
    existing = {
        (item.get("ticker"), item.get("signal_date"), item.get("target_date"), item.get("signal"))
        for item in memory.get("patterns", [])
    }
    if key not in existing:
        memory.setdefault("patterns", []).append(record)
        save_json_file(PATTERN_MEMORY_FILE, memory)
        log_pattern_to_db(record)


def nearest_close_on_or_after(df, target_date):
    if df.empty:
        return None, None
    target = pd.Timestamp(target_date)
    future = df[df.index >= target]
    if future.empty:
        return None, None
    return future.index[0], float(future["Close"].iloc[0])


def check_paper_predictions():
    section("PAPER TRADING CHECK")
    if not PAPER_TRADES_FILE.exists():
        print("  No paper trading journal yet. Use menu option 5 first.")
        return

    journal = pd.read_csv(PAPER_TRADES_FILE)
    if journal.empty:
        print("  Paper trading journal is empty.")
        return

    updated = 0
    today = pd.Timestamp.today().normalize()
    for idx, row in journal.iterrows():
        if str(row.get("status", "OPEN")) == "CLOSED":
            continue
        target_date = pd.Timestamp(row["target_date"])
        if target_date > today:
            continue

        df = fetch(row["ticker"], period="1y")
        exit_date, exit_price = nearest_close_on_or_after(df, target_date)
        if exit_price is None:
            continue

        entry_price = float(row["entry_price"])
        actual_return = exit_price / entry_price - 1
        if row["signal"] == "DOWN":
            correct = actual_return < 0
        elif row["signal"] == "UP":
            correct = actual_return > 0
        else:
            correct = abs(actual_return) < 0.005

        journal.loc[idx, "checked_at"] = dt.datetime.now().isoformat()
        journal.loc[idx, "exit_date"] = exit_date.date().isoformat()
        journal.loc[idx, "exit_price"] = exit_price
        journal.loc[idx, "actual_return"] = actual_return
        journal.loc[idx, "correct"] = bool(correct)
        journal.loc[idx, "status"] = "CLOSED"
        append_pattern_memory(journal.loc[idx].to_dict())
        updated += 1

    journal.to_csv(PAPER_TRADES_FILE, index=False)
    closed = journal[journal["status"] == "CLOSED"].copy()
    open_count = int((journal["status"] != "CLOSED").sum())
    if closed.empty:
        print(f"  No paper trades ready to close yet. Open: {open_count}")
        return

    closed["actual_return"] = pd.to_numeric(closed["actual_return"], errors="coerce")
    accuracy = closed["correct"].astype(str).str.lower().eq("true").mean() * 100
    avg_return = closed["actual_return"].mean() * 100
    print(f"  Updated closed trades : {updated}")
    print(f"  Closed / Open         : {len(closed)} / {open_count}")
    print(f"  Paper accuracy        : {accuracy:.1f}%")
    print(f"  Avg actual return     : {avg_return:+.2f}%")

    recent = closed.tail(12)
    rows = []
    for _, row in recent.iterrows():
        icon = "YES" if str(row["correct"]).lower() == "true" else "NO"
        rows.append(
            [
                row["signal_date"],
                row["ticker"],
                row["signal"],
                row["quality"],
                f"{float(row['prob_up']) * 100:.1f}%",
                f"{float(row['actual_return']) * 100:+.2f}%",
                icon,
            ]
        )
    print(tabulate(rows, headers=["Date", "Ticker", "Signal", "Quality", "Prob Up", "Return", "Correct"], tablefmt="simple"))


def pattern_memory_report():
    section("PATTERN MEMORY")
    memory = load_json_file(PATTERN_MEMORY_FILE, {"patterns": []})
    patterns = memory.get("patterns", [])
    if not patterns:
        print("  No closed pattern records yet. Let paper trades close first.")
        return

    df = pd.DataFrame(patterns)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["correct"] = df["correct"].astype(bool)

    print(f"  Stored closed patterns: {len(df)}")
    print(f"  Overall accuracy      : {df['correct'].mean() * 100:.1f}%")
    print(f"  Avg result            : {df['result'].mean() * 100:+.2f}%")

    tag_rows = []
    tag_stats = {}
    for _, row in df.iterrows():
        for tag in row.get("tags", []):
            stats = tag_stats.setdefault(tag, {"total": 0, "correct": 0, "returns": []})
            stats["total"] += 1
            stats["correct"] += int(bool(row["correct"]))
            stats["returns"].append(row["result"])

    for tag, stats in tag_stats.items():
        if stats["total"] < 2:
            continue
        avg_return = np.nanmean(stats["returns"]) * 100
        accuracy = stats["correct"] / stats["total"] * 100
        tag_rows.append([tag, stats["total"], f"{accuracy:.1f}%", f"{avg_return:+.2f}%"])

    if tag_rows:
        tag_rows = sorted(tag_rows, key=lambda row: float(row[3].replace("%", "")), reverse=True)
        section("BEST/WORST TAGS")
        print(tabulate(tag_rows[:10], headers=["Tag", "Count", "Accuracy", "Avg Return"], tablefmt="simple"))
        if len(tag_rows) > 10:
            print("\n  Weakest tags:")
            print(tabulate(tag_rows[-10:], headers=["Tag", "Count", "Accuracy", "Avg Return"], tablefmt="simple"))

    setup_rows = []
    for setup, group in df.groupby("setup"):
        if len(group) < 2:
            continue
        setup_rows.append(
            [
                setup[:55],
                len(group),
                f"{group['correct'].mean() * 100:.1f}%",
                f"{group['result'].mean() * 100:+.2f}%",
            ]
        )
    if setup_rows:
        setup_rows = sorted(setup_rows, key=lambda row: float(row[3].replace("%", "")), reverse=True)
        section("BEST SETUP COMBINATIONS")
        print(tabulate(setup_rows[:10], headers=["Setup", "Count", "Accuracy", "Avg Return"], tablefmt="simple"))


def robustness_tester(tickers):
    section("ROBUSTNESS TESTER")
    print("  Testing a compact grid. This can take a few minutes because it rebuilds walk-forward datasets.")
    period = input(f"  Lookback period [{DEFAULT_LOOKBACK}]: ").strip() or DEFAULT_LOOKBACK
    broaden = input("  Broaden with diversified liquid ETFs/stocks? [Y/n]: ").strip().lower() in ("", "y")
    if broaden:
        tickers = sorted(set([t.upper() for t in tickers] + BROAD_ROBUSTNESS_TICKERS))
        print(f"  Broadened universe: {len(tickers)} tickers")
    min_coverage = ask_float("Minimum coverage percent", DEFAULT_MIN_COVERAGE)
    mode_raw = input(f"  Training mode: setups or all [{DEFAULT_TRAINING_MODE}]: ").strip().lower()
    training_mode = mode_raw if mode_raw in ("setups", "all") else DEFAULT_TRAINING_MODE
    fee_bps = ask_int("Estimated fee/slippage in basis points", DEFAULT_FEE_BPS)
    stress_slippage_bps = ask_int("Stress slippage/gap penalty in basis points", DEFAULT_STRESS_SLIPPAGE_BPS)
    max_position_size = ask_float("Max position size, e.g. 0.15 = 15% size", DEFAULT_MAX_POSITION_SIZE)
    max_portfolio_exposure = ask_float("Max total portfolio exposure, e.g. 0.25 = 25%", DEFAULT_MAX_PORTFOLIO_EXPOSURE)
    max_sector_exposure = ask_float("Max sector exposure, e.g. 0.12 = 12%", DEFAULT_MAX_SECTOR_EXPOSURE)
    daily_stop = ask_float("Daily portfolio stop, e.g. 0.008 = 0.8%", DEFAULT_DAILY_STOP)
    equity_breaker = ask_float("Equity breaker, e.g. 0.12 = stop after -12% DD", DEFAULT_EQUITY_BREAKER)
    strict_filter = input("  Use strict whale/regime filter? [Y/n]: ").strip().lower() in ("", "y")

    horizons = [5, 10, 20]
    move_thresholds = [0.02, 0.03]
    risk_configs = [
        ("fixed", 0.04, 0.08),
        ("fixed", 0.06, 0.12),
        ("atr", DEFAULT_ATR_STOP_MULT, DEFAULT_ATR_TARGET_MULT),
        ("atr", 2.0, 4.0),
    ]
    rows = []

    for horizon in horizons:
        for move_threshold in move_thresholds:
            print(f"  Dataset: horizon={horizon}d move={move_threshold:.0%}")
            data = make_dataset(tickers, period, horizon, move_threshold, training_mode)
            if data.empty:
                continue
            for risk_mode, stop_loss, take_profit in risk_configs:
                model = candidate_models()["hist_gradient_boosting"]
                metrics = walk_forward_backtest(
                    data,
                    "hist_gradient_boosting",
                    model,
                    DEFAULT_PROB_UP,
                    DEFAULT_PROB_DOWN,
                    fee_bps,
                    min_coverage,
                    stop_loss,
                    take_profit,
                    max_position_size,
                    max_portfolio_exposure,
                    max_sector_exposure,
                    daily_stop,
                    equity_breaker,
                    strict_filter,
                    "stop_first",
                    0,
                    risk_mode=risk_mode,
                    horizon_days=horizon,
                )
                if not metrics:
                    continue
                target_first_metrics = evaluate_predictions(
                    metrics["rows"].to_dict("records"),
                    fee_bps,
                    metrics["up_threshold"],
                    metrics["down_threshold"],
                    stop_loss,
                    take_profit,
                    max_position_size,
                    max_portfolio_exposure,
                    max_sector_exposure,
                    daily_stop,
                    equity_breaker,
                    strict_filter,
                    "target_first",
                    0,
                    risk_mode,
                )
                stress_metrics = evaluate_predictions(
                    metrics["rows"].to_dict("records"),
                    fee_bps,
                    metrics["up_threshold"],
                    metrics["down_threshold"],
                    stop_loss,
                    take_profit,
                    max_position_size,
                    max_portfolio_exposure,
                    max_sector_exposure,
                    daily_stop,
                    equity_breaker,
                    strict_filter,
                    "stop_first",
                    stress_slippage_bps,
                    risk_mode,
                )
                final_metrics = metrics.get("final_test_metrics") or {}
                pf = metrics["profit_factor"]
                stress_pf = stress_metrics.get("portfolio_profit_factor", 0)
                final_portfolio_pf = final_metrics.get("portfolio_profit_factor", 0)
                final_dd = final_metrics.get("max_drawdown", 0)
                robustness_score = (
                    metrics["accuracy"] * 0.35
                    + min(pf, 3) * 4
                    + min(metrics.get("portfolio_profit_factor", 0), 3) * 7
                    + min(final_portfolio_pf, 3) * 12
                    + min(stress_pf, 3) * 14
                    + metrics["avg_return"] * 2
                    + metrics["coverage"] * 0.04
                    + metrics["max_drawdown"] * 0.85
                    + final_dd * 1.10
                )
                row = {
                    "score": robustness_score,
                    "horizon": horizon,
                    "move": move_threshold,
                    "risk_mode": risk_mode,
                    "training_mode": training_mode,
                    "stop": stop_loss,
                    "target": take_profit,
                    "accuracy": metrics["accuracy"],
                    "coverage": metrics["coverage"],
                    "avg_return": metrics["avg_return"],
                    "profit_factor": pf,
                    "portfolio_profit_factor": metrics.get("portfolio_profit_factor", 0),
                    "max_drawdown": metrics["max_drawdown"],
                    "final_accuracy": final_metrics.get("accuracy", 0),
                    "final_coverage": final_metrics.get("coverage", 0),
                    "final_profit_factor": final_metrics.get("profit_factor", 0),
                    "final_portfolio_profit_factor": final_portfolio_pf,
                    "final_max_drawdown": final_dd,
                    "target_first_profit_factor": target_first_metrics.get("portfolio_profit_factor", 0),
                    "stress_profit_factor": stress_pf,
                    "stress_slippage_bps": stress_slippage_bps,
                    "down_threshold": metrics["down_threshold"],
                    "up_threshold": metrics["up_threshold"],
                    "strict_filter": strict_filter,
                    "max_position_size": max_position_size,
                    "max_portfolio_exposure": max_portfolio_exposure,
                    "max_sector_exposure": max_sector_exposure,
                    "daily_stop": daily_stop,
                    "equity_breaker": equity_breaker,
                }
                rows.append(row)
                log_robustness_to_db(row)

    if not rows:
        print(f"  {Fore.YELLOW}No robustness results produced.{Style.RESET_ALL}")
        return

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_path = ROOT / "robustness_results.csv"
    results.to_csv(out_path, index=False)

    def fmt_pf(value):
        return "inf" if np.isinf(value) else f"{value:.2f}"

    def risk_text(row):
        if row.get("risk_mode") == "atr":
            return f"ATR {row['stop']:.1f}x/{row['target']:.1f}x"
        return f"{row['stop']:.0%}/{row['target']:.0%}"

    display = []
    for _, row in results.head(12).iterrows():
        display.append(
            [
                f"{int(row['horizon'])}d",
                f"{row['move']:.0%}",
                risk_text(row),
                f"{row['accuracy']:.1f}%",
                f"{row['coverage']:.1f}%",
                f"{row['avg_return']:+.2f}%",
                fmt_pf(row["profit_factor"]),
                fmt_pf(row["portfolio_profit_factor"]),
                f"{row['max_drawdown']:.1f}%",
                fmt_pf(row["final_portfolio_profit_factor"]),
                f"{row['final_max_drawdown']:.1f}%",
                fmt_pf(row["target_first_profit_factor"]),
                fmt_pf(row["stress_profit_factor"]),
                f"{row['down_threshold']:.2f}/{row['up_threshold']:.2f}",
                f"{row['max_position_size']:.2f}x",
                f"{row['max_portfolio_exposure']:.2f}x",
                f"{row['max_sector_exposure']:.2f}x",
            ]
        )
    print(
        tabulate(
            display,
            headers=[
                "Horizon",
                "Move",
                "Stop/TP",
                "Acc",
                "Coverage",
                "Avg",
                "Trade PF",
                "Port PF",
                "Max DD",
                "Final Port PF",
                "Final DD",
                "Target-First PF",
                "Stress PF",
                "Thresh",
                "Max Size",
                "Max Exp",
                "Sector Cap",
            ],
            tablefmt="simple",
        )
    )
    print(f"  {Fore.GREEN}Full robustness results saved -> {out_path}{Style.RESET_ALL}")


def show_memory(mem):
    section("MODEL MEMORY")
    init_db()
    print(f"  SQLite database -> {DB_FILE}")
    if not mem.get("runs"):
        print("  No completed training runs yet.")
        return
    rows = []
    for run in mem["runs"][-10:]:
        rows.append(
            [
                run["trained_at"][:10],
                run["name"],
                f"{run['accuracy']:.1f}%",
                f"{run['coverage']:.1f}%",
                run["profit_factor"],
                f"{run.get('prob_down', 0):.2f}/{run.get('prob_up', 0):.2f}",
                f"{run.get('stop_loss', 0):.0%}/{run.get('take_profit', 0):.0%}",
                f"{run['avg_return']:+.2f}%",
                f"{run['max_drawdown']:.1f}%",
            ]
        )
    print(
        tabulate(
            rows,
            headers=["Date", "Model", "Accuracy", "Coverage", "PF", "Thresholds", "Stop/TP", "Avg", "Max DD"],
            tablefmt="simple",
        )
    )


def print_banner():
    print(f"\n{Fore.CYAN}{'=' * 72}")
    print("  SMART MARKET AI - Walk-Forward ML Predictor")
    print("  Whale footprints | Market regime | Ensemble ML | Risk-managed backtests")
    print(f"{'=' * 72}{Style.RESET_ALL}")


def get_tickers():
    print(f"\n  Default tickers: {', '.join(DEFAULT_TICKERS)}")
    print("  1) Use default AI/construction basket")
    print("  2) Enter custom comma-separated tickers")
    print("  3) Build dynamic universe from screener")
    choice = input("  Ticker source [1-3]: ").strip()
    if choice in ("", "1"):
        return DEFAULT_TICKERS
    if choice == "3":
        size = ask_int("Universe size", DEFAULT_UNIVERSE_SIZE)
        size = max(5, min(size, 50))
        return build_dynamic_universe(size)

    custom = input("  Enter up to 50 comma-separated tickers: ").strip()
    tickers = [clean_symbol(t) for t in custom.split(",") if t.strip()]
    tickers = [ticker for ticker in tickers if ticker]
    return tickers[:50] or DEFAULT_TICKERS


def ask_float(prompt, default):
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def ask_int(prompt, default):
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def main():
    load_env_file()
    print_banner()
    mem = load_memory()
    if mem.get("last_run"):
        print(f"  Memory loaded | Last run: {mem['last_run'][:10]}")
    else:
        print("  Fresh ML memory.")

    tickers = get_tickers()

    while True:
        section("MAIN MENU")
        print("  1) Train + walk-forward backtest")
        print("  2) Show live predictions")
        print("  3) Show top setups only")
        print("  4) Save paper-trading predictions")
        print("  5) Check paper-trading results")
        print("  6) Run robustness tester")
        print("  7) Show model memory")
        print("  8) Market intelligence report")
        print("  9) Bearish watchlist")
        print("  10) Pattern memory report")
        print("  11) Change/rebuild ticker universe")
        print("  12) Exit")
        choice = input(f"\n  {Fore.WHITE}Choice [1-12]: {Style.RESET_ALL}").strip()

        if choice == "1":
            period = input(f"  Lookback period [{DEFAULT_LOOKBACK}]: ").strip() or DEFAULT_LOOKBACK
            horizon = ask_int("Prediction horizon in trading days", DEFAULT_HORIZON_DAYS)
            ret_threshold = ask_float("Meaningful move threshold, e.g. 0.02 = 2%", DEFAULT_RETURN_THRESHOLD)
            mode_raw = input(f"  Training mode: setups or all [{DEFAULT_TRAINING_MODE}]: ").strip().lower()
            training_mode = mode_raw if mode_raw in ("setups", "all") else DEFAULT_TRAINING_MODE
            labeling_raw = input("  Labeling: fixed or triple_barrier [fixed]: ").strip().lower()
            labeling = labeling_raw if labeling_raw in ("fixed", "triple_barrier") else "fixed"
            up_threshold = ask_float("Starting UP threshold for comparison", DEFAULT_PROB_UP)
            down_threshold = ask_float("Starting DOWN threshold for comparison", DEFAULT_PROB_DOWN)
            min_coverage = ask_float("Minimum trade coverage for auto-threshold tuning, percent", DEFAULT_MIN_COVERAGE)
            fee_bps = ask_int("Estimated fee/slippage in basis points per trade", DEFAULT_FEE_BPS)
            stop_loss = ask_float("Stop loss, e.g. 0.04 = 4%", DEFAULT_STOP_LOSS)
            take_profit = ask_float("Take profit, e.g. 0.08 = 8%", DEFAULT_TAKE_PROFIT)
            max_position_size = ask_float("Max position size, e.g. 0.15 = 15% size", DEFAULT_MAX_POSITION_SIZE)
            max_portfolio_exposure = ask_float("Max total portfolio exposure, e.g. 0.25 = 25%", DEFAULT_MAX_PORTFOLIO_EXPOSURE)
            max_sector_exposure = ask_float("Max sector exposure, e.g. 0.12 = 12%", DEFAULT_MAX_SECTOR_EXPOSURE)
            daily_stop = ask_float("Daily portfolio stop, e.g. 0.008 = 0.8%", DEFAULT_DAILY_STOP)
            equity_breaker = ask_float("Equity breaker, e.g. 0.12 = stop after -12% DD", DEFAULT_EQUITY_BREAKER)
            train_and_backtest(
                tickers,
                mem,
                period,
                horizon,
                ret_threshold,
                training_mode,
                up_threshold,
                down_threshold,
                min_coverage,
                fee_bps,
                stop_loss,
                take_profit,
                max_position_size,
                max_portfolio_exposure,
                max_sector_exposure,
                daily_stop,
                equity_breaker,
                labeling=labeling,
            )
        elif choice == "2":
            live_predict(tickers)
        elif choice == "3":
            top_setups_only(tickers)
        elif choice == "4":
            top_only = input("  Save only top setups? [Y/n]: ").strip().lower() in ("", "y")
            save_paper_predictions(tickers, top_only=top_only)
        elif choice == "5":
            check_paper_predictions()
        elif choice == "6":
            robustness_tester(tickers)
        elif choice == "7":
            show_memory(mem)
        elif choice == "8":
            market_intelligence_report(tickers)
        elif choice == "9":
            bearish_watchlist(tickers)
        elif choice == "10":
            pattern_memory_report()
        elif choice == "11":
            tickers = get_tickers()
            print(f"  Using: {Fore.YELLOW}{', '.join(tickers)}{Style.RESET_ALL}")
        elif choice == "12":
            save_memory(mem)
            print(f"\n  {Fore.CYAN}Goodbye. Memory saved.{Style.RESET_ALL}\n")
            break
        else:
            print(f"  {Fore.YELLOW}Invalid choice.{Style.RESET_ALL}")


register_model_namespace()

if __name__ == "__main__":
    main()