from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


def download_mlb_odds() -> Path | None:
    """Download consensus MLB moneyline odds when ODDS_API_KEY is configured.

    The pipeline remains operational without a key and will label every game NO BET.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not configured; odds step skipped")
        return None

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = requests.get(ODDS_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    rows: list[dict] = []
    for event in payload:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {outcome.get("name"): outcome.get("price") for outcome in market.get("outcomes", [])}
                rows.append(
                    {
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker.get("key"),
                        "bookmaker_title": bookmaker.get("title"),
                        "last_update": bookmaker.get("last_update"),
                        "home_decimal_odds": prices.get(home_team),
                        "away_decimal_odds": prices.get(away_team),
                    }
                )

    output = Path("data/raw/mlb/mlb_moneyline_odds.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"MLB odds saved: {output} ({len(rows):,} bookmaker rows)")
    return output
