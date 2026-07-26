from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

RETROSHEET_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _download_retrosheet_year(year: int) -> pd.DataFrame:
    response = requests.get(RETROSHEET_URL.format(year=year), timeout=90)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        txt_name = next(name for name in archive.namelist() if name.lower().endswith(".txt"))
        raw = pd.read_csv(archive.open(txt_name), header=None, low_memory=False)

    games = raw.iloc[:, [0, 3, 6, 9, 10]].copy()
    games.columns = ["date", "away_team", "home_team", "away_runs", "home_runs"]
    games["date"] = pd.to_datetime(games["date"].astype(str), format="%Y%m%d", errors="coerce")
    games["season"] = year
    games["away_runs"] = pd.to_numeric(games["away_runs"], errors="coerce")
    games["home_runs"] = pd.to_numeric(games["home_runs"], errors="coerce")
    return games.dropna(subset=["date", "away_team", "home_team", "away_runs", "home_runs"])


def _download_current_schedule(days_ahead: int = 2) -> pd.DataFrame:
    start = date.today()
    end = start + timedelta(days=days_ahead)
    params = {
        "sportId": 1,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "hydrate": "probablePitcher,team",
    }
    response = requests.get(MLB_SCHEDULE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    rows: list[dict] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            rows.append(
                {
                    "game_date": date_block.get("date"),
                    "game_pk": game.get("gamePk"),
                    "status": game.get("status", {}).get("detailedState"),
                    "away_team": away.get("team", {}).get("name"),
                    "home_team": home.get("team", {}).get("name"),
                    "away_pitcher": away.get("probablePitcher", {}).get("fullName"),
                    "home_pitcher": home.get("probablePitcher", {}).get("fullName"),
                }
            )
    return pd.DataFrame(rows)


def download_mlb_data(start_year: int = 2018) -> tuple[Path, Path]:
    target = Path("data/raw/mlb")
    target.mkdir(parents=True, exist_ok=True)

    latest_complete_year = date.today().year - 1
    frames = []
    for year in range(start_year, latest_complete_year + 1):
        try:
            frames.append(_download_retrosheet_year(year))
            print(f"Retrosheet {year}: OK")
        except Exception as exc:
            print(f"Retrosheet {year}: skipped ({exc})")

    if not frames:
        raise RuntimeError("No MLB historical data could be downloaded")

    history = pd.concat(frames, ignore_index=True).sort_values("date")
    history_path = target / "mlb_games_history.csv"
    history.to_csv(history_path, index=False)

    schedule = _download_current_schedule()
    schedule_path = target / "mlb_upcoming_schedule.csv"
    schedule.to_csv(schedule_path, index=False)

    print(f"MLB history saved: {history_path} ({len(history):,} games)")
    print(f"MLB schedule saved: {schedule_path} ({len(schedule):,} games)")
    return history_path, schedule_path
