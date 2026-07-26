from pathlib import Path
from datetime import datetime, timezone

from src.download_mlb import download_mlb_data
from src.mlb_model import generate_predictions, train_and_backtest


def ensure_directories() -> None:
    for folder in ["data/raw", "data/processed", "models", "backtests", "reports", "predictions"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    started = datetime.now(timezone.utc)
    print(f"Pipeline started: {started.isoformat()}")

    history_path, schedule_path = download_mlb_data()
    ratings, metrics = train_and_backtest(history_path)
    prediction_path = generate_predictions(schedule_path, ratings)

    print(f"MLB backtest: {metrics}")
    print(f"MLB predictions: {prediction_path}")
    finished = datetime.now(timezone.utc)
    print(f"Pipeline finished: {finished.isoformat()}")


if __name__ == "__main__":
    main()
