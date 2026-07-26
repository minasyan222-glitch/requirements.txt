from pathlib import Path


def download_tennis_data() -> None:
    """Prepare tennis data directory.

    The next version will ingest ATP/WTA match and odds archives.
    """
    target = Path("data/raw/tennis")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Tennis data directory ready: {target}")
