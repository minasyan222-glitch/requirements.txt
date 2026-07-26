from pathlib import Path


def download_football_data() -> None:
    """Prepare football data directory.

    The next version will ingest historical results, odds and xG data.
    """
    target = Path("data/raw/football")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Football data directory ready: {target}")
