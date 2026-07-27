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
MIN_EDGE = 0.035
MIN_EV = 0.025
MAX_KELLY_UNITS = 1.0
FRACTIONAL_KELLY = 0.25

TEAM_ALIASES = {
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


def _win_probability(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_rating + HOME_ADVANTAGE - away_rating) / 400.0))


def _update_ratings(home_rating: float, away_rating: float, home_win: float, probability: float) -> tuple[float, float]:
    change = K_FACTOR * (home_win - probability)
    return home_rating + change, away_rating - change


def _pitcher_probability_adjustment(row) -> tuple[float, str]:
    """Return a small probability adjustment based on confirmed starter quality.

    This is deliberately capped because current-season ERA/WHIP are noisy and the
    baseline has not yet been backtested with historical starter data.
    """
    home_era = pd.to_numeric(getattr(row, "home_pitcher_era", None), errors="coerce")
    away_era = pd.to_numeric(getattr(row, "away_pitcher_era", None), errors="coerce")
    home_whip = pd.to_numeric(getattr(row, "home_pitcher_whip", None), errors="coerce")
    away_whip = pd.to_numeric(getattr(row, "away_pitcher_whip", None), errors="coerce")
    home_ip = pd.to_numeric(getattr(row, "home_pitcher_innings_pitched", None), errors="coerce")
    away_ip = pd.to_numeric(getattr(row, "away_pitcher_innings_pitched", None), errors="coerce")

    required = [home_era, away_era, home_whip, away_whip, home_ip, away_ip]
    if any(pd.isna(value) for value in required):
        return 0.0, "low"
    if home_ip < 20 or away_ip < 20:
        return 0.0, "low"

    era_component = (away_era - home_era) * 0.012
    whip_component = (away_whip - home_whip) * 0.05
    adjustment = max(-0.06, min(0.06, era_component + whip_component))
    certainty = "high" if home_ip >= 60 and away_ip >= 60 else "medium"
    return float(adjustment), certainty


def _consensus_odds(odds_path: Path | None) -> pd.DataFrame:
    if odds_path is None or not odds_path.exists():
        return pd.DataFrame()
    odds = pd.read_csv(odds_path)
    if odds.empty:
        return odds
    return (
        odds.groupby(["home_team", "away_team"], as_index=False)
        .agg(
            home_decimal_odds=("home_decimal_odds", "median"),
            away_decimal_odds=("away_decimal_odds", "median"),
            bookmakers=("bookmaker", "nunique"),
        )
    )


def _kelly_units(probability: float, decimal_odds: float) -> float:
    if not decimal_odds or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    full_kelly = (b * probability - (1.0 - probability)) / b
    return round(max(0.0, min(MAX_KELLY_UNITS, full_kelly * FRACTIONAL_KELLY)), 2)


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
        "note": "Pitcher adjustments and betting ROI require a separate historical feature/odds backtest and are not included in these baseline metrics.",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/mlb_backtest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path("models").mkdir(exist_ok=True)
    Path("models/mlb_elo_ratings.json").write_text(json.dumps(dict(ratings), indent=2), encoding="utf-8")
    return dict(ratings), metrics


def generate_predictions(schedule_path: Path, ratings: dict[str, float], odds_path: Path | None = None) -> Path:
    schedule = pd.read_csv(schedule_path)
    odds = _consensus_odds(odds_path)
    if not odds.empty:
        schedule = schedule.merge(odds, on=["home_team", "away_team"], how="left")

    rows = []
    for row in schedule.itertuples(index=False):
        home_code = TEAM_ALIASES.get(row.home_team, row.home_team)
        away_code = TEAM_ALIASES.get(row.away_team, row.away_team)
        elo_probability = _win_probability(ratings.get(home_code, BASE_RATING), ratings.get(away_code, BASE_RATING))
        pitcher_adjustment, lineup_certainty = _pitcher_probability_adjustment(row)
        raw_probability = max(0.08, min(0.92, elo_probability + pitcher_adjustment))

        uncertainty_penalty = 0.012 if lineup_certainty == "low" else 0.006 if lineup_certainty == "medium" else 0.0
        home_probability = 0.5 + (raw_probability - 0.5) * (1.0 - uncertainty_penalty * 10)
        away_probability = 1.0 - home_probability

        home_odds = pd.to_numeric(getattr(row, "home_decimal_odds", None), errors="coerce")
        away_odds = pd.to_numeric(getattr(row, "away_decimal_odds", None), errors="coerce")
        candidates = []
        for side, probability, decimal_odds in (
            ("HOME", home_probability, home_odds),
            ("AWAY", away_probability, away_odds),
        ):
            if pd.isna(decimal_odds) or decimal_odds <= 1:
                continue
            implied = 1.0 / decimal_odds
            edge = probability - implied
            ev = probability * decimal_odds - 1.0
            candidates.append((side, probability, float(decimal_odds), edge, ev))

        eligible = [item for item in candidates if item[3] >= MIN_EDGE and item[4] >= MIN_EV]
        if eligible:
            side, probability, decimal_odds, edge, ev = max(eligible, key=lambda item: item[4])
            recommendation = f"BET {side}"
            stake_units = _kelly_units(probability, decimal_odds)
        else:
            recommendation = "NO BET"
            stake_units = 0.0
            side = None
            decimal_odds = None
            edge = None
            ev = None

        rows.append({
            "game_date": row.game_date,
            "away_team": row.away_team,
            "home_team": row.home_team,
            "away_pitcher": getattr(row, "away_pitcher", None),
            "home_pitcher": getattr(row, "home_pitcher", None),
            "elo_home_probability": round(elo_probability, 4),
            "pitcher_adjustment": round(pitcher_adjustment, 4),
            "lineup_certainty": lineup_certainty,
            "home_win_probability": round(home_probability, 4),
            "away_win_probability": round(away_probability, 4),
            "fair_home_decimal_odds": round(1 / home_probability, 3),
            "fair_away_decimal_odds": round(1 / away_probability, 3),
            "market_home_decimal_odds": None if pd.isna(home_odds) else round(float(home_odds), 3),
            "market_away_decimal_odds": None if pd.isna(away_odds) else round(float(away_odds), 3),
            "recommendation": recommendation,
            "selected_side": side,
            "selected_odds": decimal_odds,
            "edge": None if edge is None else round(edge, 4),
            "ev": None if ev is None else round(ev, 4),
            "stake_units": stake_units,
        })

    output = Path("predictions/mlb_predictions.csv")
    output.parent.mkdir(exist_ok=True)
    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        ranked = predictions[predictions["recommendation"] != "NO BET"].sort_values("ev", ascending=False)
        selected_indices = set(ranked.head(3).index)
        predictions.loc[
            (predictions["recommendation"] != "NO BET") & (~predictions.index.isin(selected_indices)),
            ["recommendation", "stake_units"],
        ] = ["NO BET — daily limit", 0.0]
    predictions.to_csv(output, index=False)
    return output
