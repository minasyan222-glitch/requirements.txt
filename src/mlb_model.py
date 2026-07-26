from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 35.0


def _win_probability(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_rating + HOME_ADVANTAGE - away_rating) / 400.0))


def _update_ratings(home_rating: float, away_rating: float, home_win: float, probability: float) -> tuple[float, float]:
    change = K_FACTOR * (home_win - probability)
    return home_rating + change, away_rating - change


def train_and_backtest(history_path: Path) -> tuple[dict[str, float], dict]:
    games = pd.read_csv(history_path, parse_dates=["date"]).sort_values("date")
    ratings: defaultdict[str, float] = defaultdict(lambda: BASE_RATING)
    predictions: list[dict] = []

    for row in games.itertuples(index=False):
        home_rating = ratings[row.home_team]
        away_rating = ratings[row.away_team]
        probability = _win_probability(home_rating, away_rating)
        home_win = 1.0 if row.home_runs > row.away_runs else 0.0
        predictions.append(
            {
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_win_probability": probability,
                "actual_home_win": home_win,
                "brier": (probability - home_win) ** 2,
                "log_loss": -(home_win * math.log(max(probability, 1e-12)) + (1 - home_win) * math.log(max(1 - probability, 1e-12))),
                "correct": float((probability >= 0.5) == bool(home_win)),
            }
        )
        ratings[row.home_team], ratings[row.away_team] = _update_ratings(
            home_rating, away_rating, home_win, probability
        )

    results = pd.DataFrame(predictions)
    Path("backtests").mkdir(exist_ok=True)
    results.to_csv("backtests/mlb_elo_game_predictions.csv", index=False)

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "games": int(len(results)),
        "accuracy": round(float(results["correct"].mean()), 4),
        "brier_score": round(float(results["brier"].mean()), 4),
        "log_loss": round(float(results["log_loss"].mean()), 4),
        "model": "Elo baseline",
        "home_advantage_points": HOME_ADVANTAGE,
        "k_factor": K_FACTOR,
        "note": "This baseline estimates win probability only. Betting ROI requires historical bookmaker odds and is not yet calculated.",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/mlb_backtest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path("models").mkdir(exist_ok=True)
    Path("models/mlb_elo_ratings.json").write_text(json.dumps(dict(ratings), indent=2), encoding="utf-8")
    return dict(ratings), metrics


def generate_predictions(schedule_path: Path, ratings: dict[str, float]) -> Path:
    schedule = pd.read_csv(schedule_path)
    aliases = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS", "Chicago White Sox": "CHA", "Chicago Cubs": "CHN",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
        "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCA",
        "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN", "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Yankees": "NYA",
        "New York Mets": "NYN", "Athletics": "OAK", "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN", "San Francisco Giants": "SFN",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN", "Tampa Bay Rays": "TBA",
        "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
    }
    rows = []
    for row in schedule.itertuples(index=False):
        home_code = aliases.get(row.home_team, row.home_team)
        away_code = aliases.get(row.away_team, row.away_team)
        probability = _win_probability(ratings.get(home_code, BASE_RATING), ratings.get(away_code, BASE_RATING))
        rows.append({
            "game_date": row.game_date,
            "away_team": row.away_team,
            "home_team": row.home_team,
            "away_pitcher": getattr(row, "away_pitcher", None),
            "home_pitcher": getattr(row, "home_pitcher", None),
            "home_win_probability": round(probability, 4),
            "away_win_probability": round(1 - probability, 4),
            "fair_home_decimal_odds": round(1 / probability, 3),
            "fair_away_decimal_odds": round(1 / (1 - probability), 3),
            "recommendation": "NO BET — bookmaker odds not supplied",
        })
    output = Path("predictions/mlb_predictions.csv")
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
