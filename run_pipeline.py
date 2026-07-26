from pathlib import Path
from datetime import datetime, timezone

from src.download_football import download_football_data
from src.download_mlb import download_mlb_data
from src.download_tennis import download_tennis_data


def ensure_directories() -> None:
    for folder in ["data/raw", "data/processed", "models", "backtests", "reports", "predictions"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    started = datetime.now(timezone.utc)
    print(f"Pipeline started: {started.isoformat()}")

    download_mlb_data()
    download_football_data()
    download_tennis_data()

    finished = datetime.now(timezone.utc)
    print(f"Pipeline finished: {finished.isoformat()}")


if __name__ == "__main__":
    main()
