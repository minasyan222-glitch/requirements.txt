from pathlib import Path


def download_mlb_data() -> None:
    """Prepare MLB data directory.

    The next version will ingest Retrosheet and Baseball Savant datasets.
    """
    target = Path("data/raw/mlb")
    target.mkdir(parents=True, exist_ok=True)
    print(f"MLB data directory ready: {target}")
