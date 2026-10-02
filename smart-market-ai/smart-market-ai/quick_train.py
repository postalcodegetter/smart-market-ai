"""
SMART MARKET AI - One-command trainer.

Runs a full walk-forward training + backtest using the project's default
settings, then saves the trained model (market_model.joblib) and memory
(brain_memory_ml.json). No interactive prompts.

Optional overrides:
    python quick_train.py --tickers NVDA,MSFT --period 3y --horizon 20

Educational/informational only. Not financial advice.
"""

import argparse
import sys

import smart_market_ai_ml as ai


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Smart Market AI model with defaults.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (default: project default universe)")
    parser.add_argument("--period", type=str, default=None,
                        help="Lookback period, e.g. 3y or 10y (default: 5y)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon in trading days (default: 20)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Meaningful move threshold, e.g. 0.02 (default: 0.03)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Training mode: setups or all (default: setups)")
    parser.add_argument("--labeling", type=str, default=None,
                        help="Labeling: fixed or triple_barrier (default: fixed)")
    return parser.parse_args()


def main():
    args = parse_args()

    ai.load_env_file()

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else ai.DEFAULT_TICKERS
    )
    tickers = [t for t in tickers if ai.clean_symbol(t)]
    if not tickers:
        tickers = ai.DEFAULT_TICKERS

    mem = ai.load_memory()
    print("\nStarting walk-forward training...")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period:  {args.period or ai.DEFAULT_LOOKBACK}")
    print(f"  Horizon: {args.horizon or ai.DEFAULT_HORIZON_DAYS} days")
    print("  Press Ctrl+C at any time to abort.\n")

    result = ai.train_and_backtest(
        tickers=tickers,
        mem=mem,
        period=args.period or ai.DEFAULT_LOOKBACK,
        horizon_days=args.horizon or ai.DEFAULT_HORIZON_DAYS,
        return_threshold=args.threshold if args.threshold is not None else ai.DEFAULT_RETURN_THRESHOLD,
        training_mode=args.mode or ai.DEFAULT_TRAINING_MODE,
        up_threshold=ai.DEFAULT_PROB_UP,
        down_threshold=ai.DEFAULT_PROB_DOWN,
        min_coverage=ai.DEFAULT_MIN_COVERAGE,
        fee_bps=ai.DEFAULT_FEE_BPS,
        stop_loss=ai.DEFAULT_STOP_LOSS,
        take_profit=ai.DEFAULT_TAKE_PROFIT,
        max_position_size=ai.DEFAULT_MAX_POSITION_SIZE,
        max_portfolio_exposure=ai.DEFAULT_MAX_PORTFOLIO_EXPOSURE,
        max_sector_exposure=ai.DEFAULT_MAX_SECTOR_EXPOSURE,
        daily_stop=ai.DEFAULT_DAILY_STOP,
        equity_breaker=ai.DEFAULT_EQUITY_BREAKER,
        labeling=args.labeling or "fixed",
    )

    if result is None:
        print("\nTraining did not complete. No model was saved.")
        sys.exit(1)

    ai.save_memory(mem)
    print("\nDone. Launch the dashboard with:  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
