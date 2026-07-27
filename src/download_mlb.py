from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

RETROSHEET_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PERSON_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{person_id}/stats"


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
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})
            rows.append(
                {
                    "game_date": date_block.get("date"),
                    "game_pk": game.get("gamePk"),
                    "status": game.get("status", {}).get("detailedState"),
                    "away_team": away.get("team", {}).get("name"),
                    "home_team": home.get("team", {}).get("name"),
                    "away_pitcher": away_pitcher.get("fullName"),
                    "home_pitcher": home_pitcher.get("fullName"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "home_pitcher_id": home_pitcher.get("id"),
                }
            )
    return pd.DataFrame(rows)


def _pitcher_season_stats(person_id: int | float | None, season: int) -> dict:
    if person_id is None or pd.isna(person_id):
        return {}
    params = {"stats": "season", "group": "pitching", "season": season}
    try:
        response = requests.get(
            MLB_PERSON_STATS_URL.format(person_id=int(person_id)), params=params, timeout=30
        )
        response.raise_for_status()
        splits = response.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        stat = splits[0].get("stat", {})
        return {
            "era": pd.to_numeric(stat.get("era"), errors="coerce"),
            "whip": pd.to_numeric(stat.get("whip"), errors="coerce"),
            "innings_pitched": pd.to_numeric(stat.get("inningsPitched"), errors="coerce"),
            "strikeouts_per_9": pd.to_numeric(stat.get("strikeoutsPer9Inn"), errors="coerce"),
            "walks_per_9": pd.to_numeric(stat.get("walksPer9Inn"), errors="coerce"),
        }
    except Exception as exc:
        print(f"Pitcher stats {person_id}: skipped ({exc})")
        return {}


def _enrich_schedule_with_pitchers(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return schedule
    season = date.today().year
    cache: dict[int, dict] = {}
    for column in ("away_pitcher_id", "home_pitcher_id"):
        for value in schedule[column].dropna().unique():
            pitcher_id = int(value)
            cache[pitcher_id] = _pitcher_season_stats(pitcher_id, season)

    for side in ("away", "home"):
        id_column = f"{side}_pitcher_id"
        for metric in ("era", "whip", "innings_pitched", "strikeouts_per_9", "walks_per_9"):
            schedule[f"{side}_pitcher_{metric}"] = schedule[id_column].map(
                lambda value: cache.get(int(value), {}).get(metric) if pd.notna(value) else None
            )
    return schedule


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

    schedule = _enrich_schedule_with_pitchers(_download_current_schedule())
    schedule_path = target / "mlb_upcoming_schedule.csv"
    schedule.to_csv(schedule_path, index=False)

    print(f"MLB history saved: {history_path} ({len(history):,} games)")
    print(f"MLB schedule saved: {schedule_path} ({len(schedule):,} games)")
    return history_path, schedule_path
