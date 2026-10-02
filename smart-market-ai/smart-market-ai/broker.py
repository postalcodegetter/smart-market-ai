"""
Alpaca broker adapter for the Smart Market AI bot.

Wraps the official alpaca-py SDK. The bot talks only to this module, so a
different broker can be swapped in later by reimplementing the same methods.

Live/paper is decided by the ALPACA_LIVE env flag: 0 = paper (safe, default),
1 = real money. Real orders additionally require "live": true in bot_config.json.
"""

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"


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


def _client(paper=True):
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        raise SystemExit(
            "alpaca-py is not installed. Run: pip install alpaca-py"
        )
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise SystemExit(
            "Alpaca credentials missing. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env"
        )
    return TradingClient(key, secret, paper=paper)


def live_endpoint():
    flag = os.environ.get("ALPACA_LIVE", "0").strip().lower()
    return flag not in ("0", "false", "no")


class Broker:
    def __init__(self, live=False):
        # "live" here is the bot_config flag. Real orders need BOTH the config
        # flag AND the ALPACA_LIVE env var.
        self._paper = not live_endpoint()
        self._allow_live = bool(live)
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _client(paper=self._paper)
        return self._client

    @property
    def mode(self):
        return "LIVE" if (self._allow_live and not self._paper) else "PAPER"

    def require_live_ok(self):
        if not self._allow_live:
            raise SystemExit(
                "Refusing real orders: set \"live\": true in bot_config.json, then re-run."
            )

    # ---- account ---------------------------------------------------------
    def equity(self):
        return float(self._get_client().get_account().equity)

    def cash(self):
        return float(self._get_client().get_account().cash)

    def buying_power(self):
        return float(self._get_client().get_account().buying_power)

    # ---- clock -----------------------------------------------------------
    def market_open(self):
        try:
            return bool(self._get_client().get_clock().is_open)
        except Exception as exc:
            print(f"  [WARN] Could not check market clock: {exc}")
            return True

    # ---- orders ----------------------------------------------------------
    def place_fractional_buy(self, ticker, notional_usd):
        """Fractional market buy using a USD notional (works with small amounts)."""
        if not self._allow_live:
            self.require_live_ok()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=ticker,
            notional=round(float(notional_usd), 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self._get_client().submit_order(req)
        return order.id if order else None

    def place_short_sell(self, ticker, notional_usd, price=None):
        """Open a short via a whole-share market SELL (Alpaca has no fractional shorts).

        Uses `notional` when the account supports it, otherwise falls back to a
        whole-share qty computed from the latest price.
        """
        if not self._allow_live:
            self.require_live_ok()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        px = price if price else self.latest_price(ticker)
        if not px or px <= 0:
            print(f"  [SKIP] No price to size short for {ticker}")
            return None
        qty = max(1, int(float(notional_usd) // float(px)))
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._get_client().submit_order(req)
        return order.id if order else None

    def close(self, ticker):
        """Market-close the full position for a ticker (covers shorts too)."""
        return self._get_client().close_position(ticker)

    def flatten_all(self):
        try:
            self._get_client().close_all_positions(cancel_orders=True)
            return True
        except Exception as exc:
            print(f"  [WARN] flatten_all failed: {exc}")
            return False

    # ---- positions -------------------------------------------------------
    def positions(self):
        rows = []
        try:
            for pos in self._get_client().get_all_positions():
                rows.append(
                    {
                        "ticker": pos.symbol,
                        "qty": float(pos.qty),
                        "side": str(getattr(pos, "side", "long") or "long").lower(),
                        "avg_entry_price": float(pos.avg_entry_price),
                        "current_price": float(pos.current_price),
                        "market_value": float(pos.market_value),
                        "unrealized_pl": float(pos.unrealized_pl),
                    }
                )
        except Exception as exc:
            print(f"  [WARN] Could not load positions: {exc}")
        return rows

    def has_position(self, ticker):
        return any(pos["ticker"] == ticker for pos in self.positions())

    def position_side(self, ticker):
        for pos in self.positions():
            if pos["ticker"] == ticker:
                return pos["side"]
        return None

    def _get_md_client(self, sandbox):
        from alpaca.data.historical import StockHistoricalDataClient
        key = os.environ.get("ALPACA_API_KEY", "").strip()
        secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
        return StockHistoricalDataClient(key, secret, sandbox=sandbox)

    # ---- prices ----------------------------------------------------------
    def latest_price(self, ticker):
        from alpaca.data.requests import StockLatestTradeRequest
        last_err = None
        for sandbox in (self._paper, not self._paper):
            try:
                trades = self._get_md_client(sandbox).get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=ticker)
                )
                return float(trades[ticker].price)
            except Exception as exc:
                last_err = exc
        try:
            import smart_market_ai_ml as ai
            df = ai.fetch(ticker, period="5d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        print(f"  [WARN] No Alpaca quote for {ticker}: {last_err}")
        return None
