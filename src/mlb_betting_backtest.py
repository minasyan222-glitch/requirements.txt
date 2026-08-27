from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MIN_EDGE = 0.035
MIN_EV = 0.025
FRACTIONAL_KELLY = 0.25
MAX_STAKE_UNITS = 1.0
MAX_DAILY_BETS = 3


def _kelly_units(probability: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    full_kelly = (b * probability - (1.0 - probability)) / b
    return float(max(0.0, min(MAX_STAKE_UNITS, full_kelly * FRACTIONAL_KELLY)))


def _max_drawdown(cumulative_profit: pd.Series) -> float:
    if cumulative_profit.empty:
        return 0.0
    running_peak = cumulative_profit.cummax()
    drawdown = cumulative_profit - running_peak
    return float(drawdown.min())


def _validate_odds(odds: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "home_team",
        "away_team",
        "home_decimal_odds",
        "away_decimal_odds",
    }
    missing = sorted(required - set(odds.columns))
    if missing:
        raise ValueError(f"Historical odds file is missing columns: {missing}")

    odds = odds.copy()
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce").dt.normalize()
    odds["home_decimal_odds"] = pd.to_numeric(odds["home_decimal_odds"], errors="coerce")
    odds["away_decimal_odds"] = pd.to_numeric(odds["away_decimal_odds"], errors="coerce")
    odds = odds.dropna(subset=["date", "home_team", "away_team", "home_decimal_odds", "away_decimal_odds"])
    odds = odds[(odds["home_decimal_odds"] > 1.0) & (odds["away_decimal_odds"] > 1.0)]

    # Multiple bookmakers are collapsed to the median closing line.
    return (
        odds.groupby(["date", "home_team", "away_team"], as_index=False)
        .agg(
            home_decimal_odds=("home_decimal_odds", "median"),
            away_decimal_odds=("away_decimal_odds", "median"),
            bookmaker_count=("home_decimal_odds", "size"),
        )
    )


def run_historical_betting_backtest(
    prediction_path: Path = Path("backtests/mlb_elo_game_predictions.csv"),
    odds_path: Path = Path("data/raw/mlb/mlb_historical_odds.csv"),
) -> dict:
    reports_dir = Path("reports")
    backtests_dir = Path("backtests")
    reports_dir.mkdir(exist_ok=True)
    backtests_dir.mkdir(exist_ok=True)

    if not odds_path.exists():
        metrics = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "blocked_missing_historical_odds",
            "expected_file": str(odds_path),
            "required_columns": [
                "date",
                "home_team",
                "away_team",
                "home_decimal_odds",
                "away_decimal_odds",
            ],
            "note": "No ROI is reported until real pre-game historical bookmaker odds are supplied.",
        }
        Path("reports/mlb_betting_backtest_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return metrics

    predictions = pd.read_csv(prediction_path)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize()
    identity_keys = ["date", "home_team", "away_team"]
    ambiguous = predictions[predictions.duplicated(identity_keys, keep=False)]
    if not ambiguous.empty:
        metrics = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "blocked_ambiguous_prediction_identity",
            "prediction_rows": int(len(predictions)),
            "ambiguous_prediction_rows": int(len(ambiguous)),
            "ambiguous_game_keys": int(ambiguous[identity_keys].drop_duplicates().shape[0]),
            "note": "Historical odds are keyed by date and teams; duplicate game keys (for example, doubleheaders) cannot be matched safely without a game identifier.",
        }
        Path("reports/mlb_betting_backtest_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return metrics
    odds = _validate_odds(pd.read_csv(odds_path))
    merged = predictions.merge(
        odds,
        on=["date", "home_team", "away_team"],
        how="inner",
        validate="one_to_one",
    ).sort_values("date")

    if merged.empty:
        metrics = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "blocked_no_matching_games",
            "prediction_rows": int(len(predictions)),
            "odds_rows": int(len(odds)),
            "note": "Team names and dates must match the Retrosheet game-level data.",
        }
        Path("reports/mlb_betting_backtest_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return metrics

    rows: list[dict] = []
    for row in merged.itertuples(index=False):
        home_prob = float(row.home_win_probability)
        away_prob = 1.0 - home_prob
        home_odds = float(row.home_decimal_odds)
        away_odds = float(row.away_decimal_odds)

        raw_home_implied = 1.0 / home_odds
        raw_away_implied = 1.0 / away_odds
        overround = raw_home_implied + raw_away_implied
        market_home_prob = raw_home_implied / overround
        market_away_prob = raw_away_implied / overround

        candidates = []
        for side, model_prob, market_prob, decimal_odds in (
            ("HOME", home_prob, market_home_prob, home_odds),
            ("AWAY", away_prob, market_away_prob, away_odds),
        ):
            edge = model_prob - market_prob
            ev = model_prob * decimal_odds - 1.0
            if edge >= MIN_EDGE and ev >= MIN_EV:
                candidates.append((side, model_prob, decimal_odds, edge, ev))

        if candidates:
            side, probability, selected_odds, edge, ev = max(candidates, key=lambda x: x[4])
            stake = _kelly_units(probability, selected_odds)
        else:
            side, selected_odds, edge, ev, stake = None, None, None, None, 0.0

        actual_home_win = int(row.actual_home_win)
        won = None
        profit = 0.0
        if side is not None and stake > 0:
            won = (side == "HOME" and actual_home_win == 1) or (side == "AWAY" and actual_home_win == 0)
            profit = stake * (selected_odds - 1.0) if won else -stake

        rows.append(
            {
                "date": row.date,
                "away_team": row.away_team,
                "home_team": row.home_team,
                "home_win_probability": home_prob,
                "market_home_probability_no_vig": market_home_prob,
                "home_decimal_odds": home_odds,
                "away_decimal_odds": away_odds,
                "selected_side": side,
                "selected_odds": selected_odds,
                "edge": edge,
                "ev": ev,
                "stake_units": stake,
                "won": won,
                "profit_units": profit,
            }
        )

    results = pd.DataFrame(rows)

    # Keep at most three highest-EV bets per calendar day.
    bets = results[results["selected_side"].notna()].copy()
    if not bets.empty:
        bets["daily_rank"] = bets.groupby("date")["ev"].rank(method="first", ascending=False)
        excluded = bets["daily_rank"] > MAX_DAILY_BETS
        bets.loc[excluded, ["selected_side", "selected_odds", "edge", "ev", "stake_units", "won", "profit_units"]] = [
            None,
            None,
            None,
            None,
            0.0,
            None,
            0.0,
        ]
        results = results.drop(columns=["daily_rank"], errors="ignore")
        results = results.set_index(["date", "away_team", "home_team"])
        bets = bets.set_index(["date", "away_team", "home_team"])
        results.update(bets)
        results = results.reset_index()

    results["cumulative_profit_units"] = results["profit_units"].cumsum()
    results.to_csv(backtests_dir / "mlb_betting_backtest.csv", index=False)

    placed = results[results["stake_units"] > 0].copy()
    total_staked = float(placed["stake_units"].sum())
    total_profit = float(placed["profit_units"].sum())
    roi = total_profit / total_staked if total_staked else 0.0

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "matched_games": int(len(results)),
        "bets": int(len(placed)),
        "wins": int(placed["won"].fillna(False).sum()),
        "losses": int(len(placed) - placed["won"].fillna(False).sum()),
        "hit_rate": round(float(placed["won"].mean()), 4) if not placed.empty else 0.0,
        "average_odds": round(float(placed["selected_odds"].mean()), 3) if not placed.empty else None,
        "total_staked_units": round(total_staked, 3),
        "profit_units": round(total_profit, 3),
        "roi": round(roi, 4),
        "max_drawdown_units": round(_max_drawdown(results["cumulative_profit_units"]), 3),
        "filters": {
            "minimum_edge": MIN_EDGE,
            "minimum_ev": MIN_EV,
            "fractional_kelly": FRACTIONAL_KELLY,
            "maximum_stake_units": MAX_STAKE_UNITS,
            "maximum_daily_bets": MAX_DAILY_BETS,
        },
        "warning": "Historical results do not guarantee future profitability.",
    }
    Path("reports/mlb_betting_backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
