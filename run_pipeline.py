from pathlib import Path
from datetime import datetime, timezone

from src.download_mlb import download_mlb_data
from src.mlb_betting_backtest import run_historical_betting_backtest
from src.mlb_model import generate_predictions, train_and_backtest
from src.mlb_odds import download_mlb_odds


def ensure_directories() -> None:
    for folder in ["data/raw", "data/processed", "models", "backtests", "reports", "predictions"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    started = datetime.now(timezone.utc)
    print(f"Pipeline started: {started.isoformat()}")

    history_path, schedule_path = download_mlb_data()
    odds_path = download_mlb_odds()
    ratings, metrics = train_and_backtest(history_path)
    betting_metrics = run_historical_betting_backtest()
    prediction_path = generate_predictions(schedule_path, ratings, odds_path)

    print(f"MLB probability backtest: {metrics}")
    print(f"MLB betting backtest: {betting_metrics}")
    print(f"MLB predictions: {prediction_path}")
    finished = datetime.now(timezone.utc)
    print(f"Pipeline finished: {finished.isoformat()}")


if __name__ == "__main__":
    main()
