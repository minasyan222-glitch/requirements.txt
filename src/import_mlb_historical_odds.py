from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "christophertreasure/major-league-baseball-vegas-data"
TARGET = Path("data/raw/mlb/mlb_historical_odds.csv")
DOWNLOAD_DIR = Path("data/raw/mlb/kaggle")


def _american_to_decimal(value) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _normalize(source: Path) -> Path:
    frame = pd.read_csv(source, low_memory=False)
    cols = list(frame.columns)

    date_col = _pick(cols, ["date", "game_date", "gamedate"])
    home_col = _pick(cols, ["home_team", "home", "hometeam", "team1"])
    away_col = _pick(cols, ["away_team", "away", "awayteam", "visitor", "team2"])
    home_ml = _pick(cols, ["home_moneyline", "home_ml", "homeml", "homeclose", "home_line"])
    away_ml = _pick(cols, ["away_moneyline", "away_ml", "awayml", "awayclose", "away_line"])

    missing = [name for name, col in {
        "date": date_col, "home_team": home_col, "away_team": away_col,
        "home_moneyline": home_ml, "away_moneyline": away_ml,
    }.items() if col is None]
    if missing:
        raise ValueError(f"Unsupported odds schema; missing columns: {missing}. Found: {cols}")

    output = pd.DataFrame({
        "date": pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
        "home_team": frame[home_col].astype(str).str.strip(),
        "away_team": frame[away_col].astype(str).str.strip(),
        "home_decimal_odds": frame[home_ml].map(_american_to_decimal),
        "away_decimal_odds": frame[away_ml].map(_american_to_decimal),
    }).dropna()

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    output.drop_duplicates(["date", "home_team", "away_team"]).to_csv(TARGET, index=False)
    print(f"Historical MLB odds normalized: {TARGET} ({len(output):,} games)")
    return TARGET


def import_mlb_historical_odds() -> Path | None:
    if TARGET.exists() and TARGET.stat().st_size > 100:
        return TARGET

    local_candidates = [
        Path("data/import/oddsDataMLB.csv"),
        DOWNLOAD_DIR / "oddsDataMLB.csv",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return _normalize(candidate)

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        print("Historical odds not imported: add KAGGLE_USERNAME and KAGGLE_KEY secrets, or place oddsDataMLB.csv in data/import/")
        return None

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", DATASET,
        "-p", str(DOWNLOAD_DIR), "--force"
    ], check=True)

    archives = sorted(DOWNLOAD_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        raise RuntimeError("Kaggle download completed but no ZIP archive was found")
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(DOWNLOAD_DIR)

    candidates = list(DOWNLOAD_DIR.rglob("oddsDataMLB.csv"))
    if not candidates:
        candidates = list(DOWNLOAD_DIR.rglob("*.csv"))
    if not candidates:
        raise RuntimeError("No CSV file found in downloaded Kaggle dataset")
    return _normalize(candidates[0])
