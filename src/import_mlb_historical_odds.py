from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from statistics import median

import pandas as pd
import requests

KAGGLE_DATASET = "christophertreasure/major-league-baseball-vegas-data"
GITHUB_RELEASE_API = "https://api.github.com/repos/ArnavSaraogi/mlb-odds-scraper/releases/tags/dataset"
TARGET = Path("data/raw/mlb/mlb_historical_odds.csv")
DOWNLOAD_DIR = Path("data/raw/mlb/kaggle")

# Convert common MLB abbreviations used by sportsbooks to Retrosheet codes.
TEAM_CODES = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHW": "CHA", "CWS": "CHA", "CHA": "CHA", "CHC": "CHN", "CHN": "CHN",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KC": "KCA", "KCR": "KCA", "KCA": "KCA",
    "LAA": "ANA", "ANA": "ANA", "LAD": "LAN", "LAN": "LAN",
    "MIA": "MIA", "FLA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NYY": "NYA", "NYA": "NYA", "NYM": "NYN", "NYN": "NYN",
    "OAK": "OAK", "ATH": "OAK", "PHI": "PHI", "PIT": "PIT",
    "SD": "SDN", "SDP": "SDN", "SDN": "SDN", "SF": "SFN", "SFG": "SFN", "SFN": "SFN",
    "SEA": "SEA", "STL": "SLN", "SLN": "SLN", "TB": "TBA", "TBR": "TBA", "TBA": "TBA",
    "TEX": "TEX", "TOR": "TOR", "WSH": "WAS", "WAS": "WAS",
}


def _american_to_decimal(value) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def _team_code(team: dict | str | None) -> str | None:
    if isinstance(team, dict):
        value = team.get("shortName") or team.get("abbreviation") or team.get("fullName")
    else:
        value = team
    if not value:
        return None
    key = str(value).strip().upper()
    return TEAM_CODES.get(key, key)


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _normalize_csv(source: Path) -> Path:
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
        "home_team": frame[home_col].map(lambda x: TEAM_CODES.get(str(x).strip().upper(), str(x).strip())),
        "away_team": frame[away_col].map(lambda x: TEAM_CODES.get(str(x).strip().upper(), str(x).strip())),
        "home_decimal_odds": frame[home_ml].map(_american_to_decimal),
        "away_decimal_odds": frame[away_ml].map(_american_to_decimal),
        "source": "kaggle_major_league_baseball_vegas_data",
    }).dropna(subset=["date", "home_team", "away_team", "home_decimal_odds", "away_decimal_odds"])

    return _save(output)


def _normalize_sbr_json(source: Path) -> Path:
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict] = []
    date_groups = payload.items() if isinstance(payload, dict) else []
    for date_key, games in date_groups:
        if not isinstance(games, list):
            continue
        for game in games:
            view = game.get("gameView") or {}
            home_team = _team_code(view.get("homeTeam"))
            away_team = _team_code(view.get("awayTeam"))
            game_date = str(view.get("startDate") or date_key)[:10]
            moneylines = ((game.get("odds") or {}).get("moneyline") or [])

            closing_home: list[float] = []
            closing_away: list[float] = []
            opening_home: list[float] = []
            opening_away: list[float] = []
            books: set[str] = set()

            for book in moneylines:
                current = book.get("currentLine") or book.get("closingLine") or {}
                opening = book.get("openingLine") or {}
                ch = _american_to_decimal(current.get("homeOdds"))
                ca = _american_to_decimal(current.get("awayOdds"))
                oh = _american_to_decimal(opening.get("homeOdds"))
                oa = _american_to_decimal(opening.get("awayOdds"))
                if ch and ca:
                    closing_home.append(ch)
                    closing_away.append(ca)
                    books.add(str(book.get("sportsbook") or "unknown"))
                if oh and oa:
                    opening_home.append(oh)
                    opening_away.append(oa)

            if not (home_team and away_team and closing_home and closing_away):
                continue

            rows.append({
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_decimal_odds": median(closing_home),
                "away_decimal_odds": median(closing_away),
                "home_open_decimal_odds": median(opening_home) if opening_home else None,
                "away_open_decimal_odds": median(opening_away) if opening_away else None,
                "bookmaker_count": len(books),
                "source": "sportsbookreview_github_release",
            })

    if not rows:
        raise RuntimeError("The public SportsBookReview release contained no usable MLB moneyline rows")
    return _save(pd.DataFrame(rows))


def _save(output: pd.DataFrame) -> Path:
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    output = output.dropna(subset=["date", "home_team", "away_team", "home_decimal_odds", "away_decimal_odds"])
    output = output.drop_duplicates(["date", "home_team", "away_team"], keep="last").sort_values("date")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(TARGET, index=False)
    print(f"Historical MLB odds normalized: {TARGET} ({len(output):,} games)")
    return TARGET


def _download_public_github_release() -> Path | None:
    try:
        response = requests.get(GITHUB_RELEASE_API, timeout=30)
        response.raise_for_status()
        assets = response.json().get("assets", [])
        json_assets = [a for a in assets if str(a.get("name", "")).lower().endswith((".json", ".json.zip", ".zip"))]
        if not json_assets:
            print("Public GitHub odds release has no JSON/ZIP asset")
            return None

        asset = max(json_assets, key=lambda item: int(item.get("size") or 0))
        url = asset.get("browser_download_url")
        if not url:
            return None

        suffix = ".zip" if str(asset.get("name", "")).lower().endswith(".zip") else ".json"
        temp_dir = Path(tempfile.mkdtemp(prefix="mlb_odds_"))
        download_path = temp_dir / f"release{suffix}"
        with requests.get(url, stream=True, timeout=(30, 300)) as download:
            download.raise_for_status()
            with download_path.open("wb") as handle:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        if suffix == ".zip":
            with zipfile.ZipFile(download_path) as archive:
                archive.extractall(temp_dir)
            candidates = list(temp_dir.rglob("*.json"))
            if not candidates:
                raise RuntimeError("GitHub release ZIP contained no JSON file")
            download_path = max(candidates, key=lambda p: p.stat().st_size)
        return _normalize_sbr_json(download_path)
    except (requests.RequestException, ValueError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Public GitHub historical odds import failed: {exc}")
        return None


def import_mlb_historical_odds() -> Path | None:
    if TARGET.exists() and TARGET.stat().st_size > 100:
        return TARGET

    local_candidates = [
        Path("data/import/mlb_odds.json"),
        Path("data/import/oddsDataMLB.csv"),
        DOWNLOAD_DIR / "oddsDataMLB.csv",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return _normalize_sbr_json(candidate) if candidate.suffix.lower() == ".json" else _normalize_csv(candidate)

    # First choice: public release with 2021-04-01 through 2025-08-16,
    # multiple sportsbooks, and both opening and closing lines.
    public_result = _download_public_github_release()
    if public_result:
        return public_result

    # Fallback: Kaggle archive covering 2012-2021 closing lines.
    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        print("Historical odds not imported: public release failed; add KAGGLE credentials or place a source file in data/import/")
        return None

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
        "-p", str(DOWNLOAD_DIR), "--force"
    ], check=True)

    archives = sorted(DOWNLOAD_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        raise RuntimeError("Kaggle download completed but no ZIP archive was found")
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(DOWNLOAD_DIR)

    candidates = list(DOWNLOAD_DIR.rglob("oddsDataMLB.csv")) or list(DOWNLOAD_DIR.rglob("*.csv"))
    if not candidates:
        raise RuntimeError("No CSV file found in downloaded Kaggle dataset")
    return _normalize_csv(candidates[0])
