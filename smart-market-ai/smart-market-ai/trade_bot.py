"""
SMART MARKET AI - Auto-Trading Bot (Alpaca, long-only, fractional).

Loop:
  1. baseline equity -> 2. retrain -> 3. open top-setup UP signals
  4. monitor positions, close any that hit their stop/target
  5. stop the session when total P&L hits profit_target_pct or session_stop_pct,
     flatten everything, log the session, then repeat from step 1.

Flags:
  --config PATH     config file (default bot_config.json)
  --once            run one cycle then exit
  --dry-run         compute/log orders, never place them
  --no-retrain      skip training (uses existing market_model.joblib)

Safety:
  - Real orders require BOTH "live": true in bot_config.json AND ALPACA_LIVE=1 in .env
  - Placing a file named STOP_BOT next to the bot flattens and exits on the next check
  - Ctrl+C flattens open positions and logs the session

Educational/informational only. Not financial advice.
"""

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOURNAL_FILE = ROOT / "bot_journal.csv"
SESSIONS_FILE = ROOT / "bot_sessions.csv"
STOP_FILE = ROOT / "STOP_BOT"
DEFAULT_CONFIG = ROOT / "bot_config.json"

JOURNAL_HEADER = [
    "opened_at", "session_id", "ticker", "side", "signal", "quality", "prob_up",
    "entry_price", "qty", "notional_usd", "stop_pct", "target_pct",
    "stop_price", "target_price", "status", "closed_at", "exit_price",
    "exit_reason", "realized_pnl_usd", "realized_pnl_pct",
]
SESSION_HEADER = [
    "started_at", "ended_at", "baseline_equity", "end_equity", "pnl_usd",
    "pnl_pct", "trades_opened", "trades_closed", "reason",
]

import smart_market_ai_ml as ai
import broker as broker_mod


def banner(mode, dry_run):
    print("\n" + "=" * 72)
    print("  SMART MARKET AI - AUTO BOT")
    print(f"  Mode: {mode}   Dry-run: {dry_run}")
    print("  Long/short book | portfolio-aware selection | regime de-risking")
    print("  Educational/informational only. Not financial advice.")
    print("=" * 72 + "\n")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_journal():
    if not JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(JOURNAL_HEADER)
    if not SESSIONS_FILE.exists():
        with open(SESSIONS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SESSION_HEADER)


def append_csv(path, header, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({h: row.get(h, "") for h in header})


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def retrain(cfg):
    section = ai.section
    section("RETRAINING")
    print("  This runs a full walk-forward training pass and can take a while.")
    try:
        mem = ai.load_memory()
        ai.train_and_backtest(
            tickers=cfg["tickers"],
            mem=mem,
            period=cfg.get("retrain_period", ai.DEFAULT_LOOKBACK),
            horizon_days=cfg.get("retrain_horizon", ai.DEFAULT_HORIZON_DAYS),
            return_threshold=ai.DEFAULT_RETURN_THRESHOLD,
            training_mode=ai.DEFAULT_TRAINING_MODE,
            up_threshold=ai.DEFAULT_PROB_UP,
            down_threshold=ai.DEFAULT_PROB_DOWN,
            min_coverage=ai.DEFAULT_MIN_COVERAGE,
            fee_bps=ai.DEFAULT_FEE_BPS,
            stop_loss=ai.DEFAULT_STOP_LOSS,
            take_profit=ai.DEFAULT_TAKE_PROFIT,
            max_position_size=cfg.get("max_position_pct", ai.DEFAULT_MAX_POSITION_SIZE),
            max_portfolio_exposure=cfg.get("max_position_pct", ai.DEFAULT_MAX_PORTFOLIO_EXPOSURE),
            max_sector_exposure=ai.DEFAULT_MAX_SECTOR_EXPOSURE,
            daily_stop=ai.DEFAULT_DAILY_STOP,
            equity_breaker=ai.DEFAULT_EQUITY_BREAKER,
            labeling="fixed",
        )
        ai.save_memory(mem)
        print("  Retrain complete.")
    except Exception as exc:
        print(f"  [WARN] Retrain failed ({exc}); continuing with the existing model if present.")
        print("         Re-run Train.bat manually, or set retrain_every_cycle to false.")


def open_trades(broker, cfg, E0, session_id, dry_run):
    """Open a portfolio-aware long/short book from select_portfolio().

    Returns list of opened positions with side (LONG/SHORT) for monitoring.
    """
    records, payload = ai.live_prediction_records(cfg["tickers"])
    if not records or payload is None:
        print("  No predictions available (is a model trained?). Skipping entries.")
        return []

    picks = ai.select_portfolio(records, cfg)
    if not picks:
        print("  No qualifying long/short setups for the book right now.")
        return []

    existing = {p["ticker"] for p in broker.positions()} if not dry_run else set()
    if not dry_run and not broker.market_open():
        print("  Market closed - skipping new entries.")
        return []

    max_total = int(cfg.get("max_total_positions", cfg.get("max_open_positions", 18)))
    max_per_trade = float(cfg.get("max_usd_per_trade", 250))
    opened = []
    longs = sum(1 for p in picks if p["side"] == "LONG")
    shorts = sum(1 for p in picks if p["side"] == "SHORT")
    regime = picks[0].get("regime", "n/a")
    scale = picks[0].get("regime_scale", 1.0)
    print(f"  Book: {longs} long / {shorts} short | regime={regime} exposure_scale={scale}")

    for p in picks:
        if len(opened) + len(existing) >= max_total:
            break
        if p["ticker"] in existing:
            continue

        weight = float(p.get("weight", 0))
        notional = E0 * weight
        notional = min(notional, max_per_trade)
        if notional <= 0:
            continue

        price = p.get("price") or broker.latest_price(p["ticker"])
        if not price:
            continue

        side = p["side"]
        stop_pct = float(p["stop_loss"])
        target_pct = float(p["take_profit"])
        if side == "LONG":
            qty = notional / float(price)
            stop_price = round(float(price) * (1 - stop_pct), 2)
            target_price = round(float(price) * (1 + target_pct), 2)
        else:
            qty = max(1, int(notional // float(price)))
            notional = qty * float(price)
            stop_price = round(float(price) * (1 + stop_pct), 2)
            target_price = round(float(price) * (1 - target_pct), 2)

        order_id = None
        if not dry_run:
            if side == "LONG":
                order_id = broker.place_fractional_buy(p["ticker"], notional)
            else:
                order_id = broker.place_short_sell(p["ticker"], notional, price)
            if order_id is None:
                print(f"  [SKIP] Order failed for {p['ticker']}")
                continue
            existing.add(p["ticker"])

        row = {
            "opened_at": now_iso(),
            "session_id": session_id,
            "ticker": p["ticker"],
            "side": side,
            "signal": p["signal"],
            "quality": p["quality"],
            "prob_up": round(float(p["prob_up"]), 4),
            "entry_price": price,
            "qty": round(qty, 4),
            "notional_usd": round(notional, 2),
            "stop_pct": stop_pct,
            "target_pct": target_pct,
            "stop_price": stop_price,
            "target_price": target_price,
            "status": "DRYRUN" if dry_run else "OPEN",
            "closed_at": "",
            "exit_price": "",
            "exit_reason": "",
            "realized_pnl_usd": "",
            "realized_pnl_pct": "",
        }
        append_csv(JOURNAL_FILE, JOURNAL_HEADER, row)
        tag = "WOULD-SELLSHORT" if (dry_run and side == "SHORT") else ("WOULD-BUY" if dry_run else f"{side} ({order_id})")
        print(f"  [{tag}] {p['ticker']} {round(notional, 2)} USD  "
              f"qty={round(qty, 4)}  stop@{stop_price}  target@{target_price}")
        opened.append({
            "ticker": p["ticker"],
            "side": side,
            "entry": float(price),
            "qty": qty,
            "notional": notional,
            "stop": stop_pct,
            "target": target_pct,
            "opened_at": row["opened_at"],
        })
    print(f"  Opened {len(opened)} position(s) this cycle.")
    return opened


def mark_closed(ticker, opened, exit_price, reason, session_id):
    for info in list(opened):
        if info["ticker"] != ticker:
            continue
        info["closed"] = True
        qty = info["qty"]
        side = info.get("side", "LONG")
        if side == "LONG":
            pnl_usd = qty * (exit_price - info["entry"])
            pnl_pct = exit_price / info["entry"] - 1 if info["entry"] else 0
            stop_price = round(info["entry"] * (1 - info["stop"]), 2)
            target_price = round(info["entry"] * (1 + info["target"]), 2)
        else:
            pnl_usd = qty * (info["entry"] - exit_price)
            pnl_pct = info["entry"] / exit_price - 1 if exit_price else 0
            stop_price = round(info["entry"] * (1 + info["stop"]), 2)
            target_price = round(info["entry"] * (1 - info["target"]), 2)
        row = {
            "opened_at": info["opened_at"],
            "session_id": session_id,
            "ticker": ticker,
            "side": side,
            "signal": "UP" if side == "LONG" else "DOWN",
            "quality": "",
            "prob_up": "",
            "entry_price": info["entry"],
            "qty": round(qty, 4),
            "notional_usd": round(info["notional"], 2),
            "stop_pct": info["stop"],
            "target_pct": info["target"],
            "stop_price": stop_price,
            "target_price": target_price,
            "status": "CLOSED",
            "closed_at": now_iso(),
            "exit_price": exit_price,
            "exit_reason": reason,
            "realized_pnl_usd": round(pnl_usd, 2),
            "realized_pnl_pct": round(pnl_pct, 4),
        }
        append_csv(JOURNAL_FILE, JOURNAL_HEADER, row)
        print(f"  [CLOSE] {ticker} ({side}) {reason}  exit={exit_price}  pnl={pnl_usd:+.2f} USD")


def monitor(broker, cfg, E0, session_id, opened):
    """Watch the long/short book; end the session on profit target, stop, or
    an equity-breaker drawdown from peak equity (regime/beta de-risking)."""
    interval = max(1, cfg["check_interval_minutes"]) * 60
    target_pct = cfg["profit_target_pct"]
    stop_pct = cfg["session_stop_pct"]
    breaker_pct = float(cfg.get("equity_breaker_pct", 0)) / 100.0

    print(f"  Monitoring up to {len(opened)} position(s) every {interval // 60} min. "
          f"Session target {target_pct:+.1f}% / stop {stop_pct:+.1f}%"
          + (f" / breaker {cfg.get('equity_breaker_pct')}%" if breaker_pct else "") + ".")

    peak_equity = E0
    while True:
        if STOP_FILE.exists():
            print("  STOP_BOT file found - flattening and stopping.")
            broker.flatten_all()
            return "interrupted"

        time.sleep(interval)

        positions = {p["ticker"]: p for p in broker.positions()}
        for info in [o for o in opened if not o.get("closed")]:
            pos = positions.get(info["ticker"])
            if pos is None:
                continue
            price = broker.latest_price(info["ticker"]) or pos["current_price"]
            if price is None:
                continue
            side = info.get("side", "LONG")
            if side == "LONG":
                ret = price / info["entry"] - 1
            else:
                ret = info["entry"] / price - 1
            if ret <= -info["stop"]:
                broker.close(info["ticker"])
                mark_closed(info["ticker"], opened, price, "stop", session_id)
            elif ret >= info["target"]:
                broker.close(info["ticker"])
                mark_closed(info["ticker"], opened, price, "target", session_id)

        equity = broker.equity()
        peak_equity = max(peak_equity, equity)
        pnl_pct = (equity - E0) / E0 * 100 if E0 else 0.0
        dd_pct = (equity / peak_equity - 1) * 100 if peak_equity else 0.0
        print(f"  [{now_iso()}] equity={equity:,.2f}  session P&L={pnl_pct:+.2f}%  "
              f"peak DD={dd_pct:+.2f}%")

        if pnl_pct >= target_pct:
            broker.flatten_all()
            print(f"  PROFIT TARGET HIT ({pnl_pct:+.2f}%) - flattening session.")
            return "target_hit"
        if pnl_pct <= stop_pct:
            broker.flatten_all()
            print(f"  SESSION STOP HIT ({pnl_pct:+.2f}%) - flattening session.")
            return "stop_hit"
        if breaker_pct and dd_pct <= -breaker_pct * 100:
            broker.flatten_all()
            print(f"  EQUITY BREAKER HIT ({dd_pct:+.2f}%) - flattening session.")
            return "breaker_hit"

        remaining = [o for o in opened if not o.get("closed")]
        if not remaining:
            return "no_positions"


def log_session(started, ended, E0, end_equity, opened_count, reason):
    pnl_usd = (end_equity - E0) if end_equity else 0.0
    pnl_pct = (pnl_usd / E0 * 100) if E0 else 0.0
    closed_count = 0
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            closed_count = sum(
                1 for row in csv.DictReader(f)
                if row.get("session_id") == started and row.get("status") == "CLOSED"
            )
    row = {
        "started_at": started,
        "ended_at": ended,
        "baseline_equity": round(E0, 2),
        "end_equity": round(end_equity, 2) if end_equity else "",
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "trades_opened": opened_count,
        "trades_closed": closed_count,
        "reason": reason,
    }
    append_csv(SESSIONS_FILE, SESSION_HEADER, row)
    print(f"\n  SESSION {started} -> {reason} | "
          f"P&L {pnl_usd:+.2f} USD ({pnl_pct:+.2f}%) | "
          f"{opened_count} opened, {closed_count} closed\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Market AI auto-trading bot")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to bot_config.json")
    parser.add_argument("--once", action="store_true", help="run one cycle then exit")
    parser.add_argument("--dry-run", action="store_true", help="log orders, never place them")
    parser.add_argument("--no-retrain", action="store_true", help="skip retraining this run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dry_run = args.dry_run or bool(cfg.get("dry_run", False))

    ensure_journal()
    broker_mod.load_env_file()
    broker = broker_mod.Broker(live=bool(cfg.get("live", False)))
    banner(broker.mode, dry_run)

    if dry_run:
        print("  DRY RUN: no real orders will be placed.\n")

    session_counter = 0
    started = now_iso()
    E0 = 0.0
    try:
        while True:
            session_counter += 1
            session_id = now_iso().replace(":", "-").replace("T", "_")
            started = now_iso()
            E0 = broker.equity() if not dry_run else 10000.0
            print(f"\n===== SESSION {session_counter} [{started}] baseline={E0:,.2f} {broker.mode} =====")

            if cfg.get("retrain_every_cycle", True) and not args.no_retrain:
                retrain(cfg)

            opened = open_trades(broker, cfg, E0, session_id, dry_run)

            reason = "dry_run" if dry_run else monitor(broker, cfg, E0, session_id, opened)
            end_equity = broker.equity() if not dry_run else E0
            log_session(started, now_iso(), E0, end_equity, len(opened), reason)

            if args.once:
                print("  --once: stopping after one cycle.")
                break
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        if not dry_run:
            try:
                if broker.positions():
                    broker.flatten_all()
                    print("  Flattened open positions.")
            except Exception:
                pass
        reason = "interrupted"
        log_session(started, now_iso(), E0, broker.equity() if not dry_run else E0, 0, reason)

    print("\n  Bot stopped. Journal: bot_journal.csv | Sessions: bot_sessions.csv")


if __name__ == "__main__":
    main()
