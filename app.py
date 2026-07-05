from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="FIFA World Cup ML Predictor",
    page_icon="⚽",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "best_match_prediction_model.pkl"
FEATURES_PATH = ROOT / "models" / "model_features.pkl"

MATCH_DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "matches_with_advanced_features.csv",
    ROOT / "data" / "processed" / "advanced_match_features.csv",
]

OFFICIAL_RESULTS_PATH = ROOT / "data" / "app" / "official_results.json"
PREDICTION_HISTORY_PATH = ROOT / "data" / "app" / "prediction_history.csv"

DEFAULT_STATE = {
    "completed_matches": []
}


# ============================================================
# Bracket Definition
# ============================================================

BRACKET_MATCHES = [
    {
        "id": "r16_2_left",
        "round": "round_of_16",
        "date": "2026-07-06",
        "home_team": "Portugal",
        "away_team": "Spain",
    },
    {
        "id": "r16_2_right",
        "round": "round_of_16",
        "date": "2026-07-06",
        "home_team": "United States",
        "away_team": "Belgium",
    },
    {
        "id": "r16_3_left",
        "round": "round_of_16",
        "date": "2026-07-05",
        "home_team": "Brazil",
        "away_team": "Norway",
    },
    {
        "id": "r16_3_right",
        "round": "round_of_16",
        "date": "2026-07-05",
        "home_team": "Mexico",
        "away_team": "England",
    },
    {
        "id": "r16_4_left",
        "round": "round_of_16",
        "date": "2026-07-07",
        "home_team": "Argentina",
        "away_team": "Egypt",
    },
    {
        "id": "r16_4_right",
        "round": "round_of_16",
        "date": "2026-07-07",
        "home_team": "Switzerland",
        "away_team": "Colombia",
    },
    {
        "id": "qf_1",
        "round": "quarterfinal",
        "date": "2026-07-09",
        "home_team": "Morocco",
        "away_team": "France",
    },
    {
        "id": "qf_2",
        "round": "quarterfinal",
        "date": "2026-07-10",
        "home_source": "r16_2_left",
        "away_source": "r16_2_right",
    },
    {
        "id": "qf_3",
        "round": "quarterfinal",
        "date": "2026-07-11",
        "home_source": "r16_3_left",
        "away_source": "r16_3_right",
    },
    {
        "id": "qf_4",
        "round": "quarterfinal",
        "date": "2026-07-11",
        "home_source": "r16_4_left",
        "away_source": "r16_4_right",
    },
    {
        "id": "sf_1",
        "round": "semifinal",
        "date": "2026-07-14",
        "home_source": "qf_1",
        "away_source": "qf_2",
    },
    {
        "id": "sf_2",
        "round": "semifinal",
        "date": "2026-07-15",
        "home_source": "qf_3",
        "away_source": "qf_4",
    },
    {
        "id": "final",
        "round": "final",
        "date": "2026-07-19",
        "home_source": "sf_1",
        "away_source": "sf_2",
    },
]

INITIAL_TEAMS = sorted(
    {
        "Morocco",
        "France",
        "Portugal",
        "Spain",
        "United States",
        "Belgium",
        "Brazil",
        "Norway",
        "Mexico",
        "England",
        "Argentina",
        "Egypt",
        "Switzerland",
        "Colombia",
    }
)


# ============================================================
# Loading Helpers
# ============================================================

def find_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path

    return None


@st.cache_resource
def load_model_and_features() -> tuple[Any, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing model file: models/best_match_prediction_model.pkl"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Missing feature file: models/model_features.pkl"
        )

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


@st.cache_data
def load_matches() -> pd.DataFrame:
    match_path = find_existing_path(MATCH_DATA_CANDIDATES)

    if match_path is None:
        raise FileNotFoundError(
            "Missing processed match data. Expected either "
            "data/processed/matches_with_advanced_features.csv or "
            "data/processed/advanced_match_features.csv."
        )

    matches = pd.read_csv(match_path)
    matches["date"] = pd.to_datetime(matches["date"])

    return matches


def get_clean_default_state() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_STATE)


def load_official_state() -> dict[str, Any]:
    if not OFFICIAL_RESULTS_PATH.exists():
        return get_clean_default_state()

    with open(OFFICIAL_RESULTS_PATH, "r", encoding="utf-8") as file:
        state = json.load(file)

    if "completed_matches" not in state:
        state["completed_matches"] = []

    validate_state(state)

    return state


def load_prediction_history() -> pd.DataFrame:
    if not PREDICTION_HISTORY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "snapshot_label",
                "team",
                "quarterfinal",
                "semifinal",
                "final",
                "champion",
            ]
        )

    return pd.read_csv(PREDICTION_HISTORY_PATH)


# ============================================================
# Bracket and State Helpers
# ============================================================

def completed_matches_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    completed = {}

    for match in state.get("completed_matches", []):
        completed[match["match_id"]] = match

    return completed


def state_signature(state: dict[str, Any]) -> str:
    return json.dumps(
        state.get("completed_matches", []),
        sort_keys=True,
    )


def get_match_by_id(match_id: str) -> dict[str, Any]:
    for match in BRACKET_MATCHES:
        if match["id"] == match_id:
            return match

    raise ValueError(f"Unknown match id: {match_id}")


def resolve_team_from_source(
    source_id: str | None,
    completed: dict[str, dict[str, Any]],
    simulated_winners: dict[str, str],
) -> str | None:
    if source_id is None:
        return None

    if source_id in simulated_winners:
        return simulated_winners[source_id]

    if source_id in completed:
        return completed[source_id]["winner"]

    return None


def resolve_match_teams(
    match: dict[str, Any],
    state: dict[str, Any],
    simulated_winners: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    if simulated_winners is None:
        simulated_winners = {}

    completed = completed_matches_by_id(state)

    home_team = match.get("home_team")
    away_team = match.get("away_team")

    if home_team is None:
        home_team = resolve_team_from_source(
            match.get("home_source"),
            completed,
            simulated_winners,
        )

    if away_team is None:
        away_team = resolve_team_from_source(
            match.get("away_source"),
            completed,
            simulated_winners,
        )

    return home_team, away_team


def get_available_matches(
    state: dict[str, Any],
) -> list[tuple[dict[str, Any], str, str]]:
    completed = completed_matches_by_id(state)
    simulated_winners = {}
    available = []

    for match in BRACKET_MATCHES:
        match_id = match["id"]

        if match_id in completed:
            simulated_winners[match_id] = completed[match_id]["winner"]
            continue

        home_team, away_team = resolve_match_teams(
            match,
            state,
            simulated_winners,
        )

        if home_team is not None and away_team is not None:
            available.append((match, home_team, away_team))

    return available


def format_round_name(round_name: str) -> str:
    return round_name.replace("_", " ").title()


def format_match_label(
    match: dict[str, Any],
    home_team: str,
    away_team: str,
) -> str:
    return (
        f"{format_round_name(match['round'])}: "
        f"{home_team} vs {away_team} ({match['date']})"
    )


def validate_state(state: dict[str, Any]) -> None:
    seen_match_ids = set()
    completed = {}

    for match in BRACKET_MATCHES:
        match_id = match["id"]

        matching_records = [
            record
            for record in state.get("completed_matches", [])
            if record["match_id"] == match_id
        ]

        if len(matching_records) == 0:
            continue

        if len(matching_records) > 1:
            raise ValueError(f"Duplicate result for match_id: {match_id}")

        record = matching_records[0]

        if match_id in seen_match_ids:
            raise ValueError(f"Duplicate result for match_id: {match_id}")

        seen_match_ids.add(match_id)

        home_team, away_team = resolve_match_teams(
            match,
            {"completed_matches": list(completed.values())},
            {},
        )

        if home_team is None:
            home_team = match.get("home_team")

        if away_team is None:
            away_team = match.get("away_team")

        if home_team is not None and record["home_team"] != home_team:
            raise ValueError(
                f"{match_id} has wrong home_team. "
                f"Expected {home_team}, got {record['home_team']}."
            )

        if away_team is not None and record["away_team"] != away_team:
            raise ValueError(
                f"{match_id} has wrong away_team. "
                f"Expected {away_team}, got {record['away_team']}."
            )

        home_score = int(record["home_score"])
        away_score = int(record["away_score"])
        winner = record["winner"]

        if home_score < 0 or away_score < 0:
            raise ValueError(f"{match_id} has a negative score.")

        if winner not in {record["home_team"], record["away_team"]}:
            raise ValueError(
                f"{match_id} winner must be one of the two teams."
            )

        if home_score > away_score and winner != record["home_team"]:
            raise ValueError(
                f"{match_id} winner does not match the score."
            )

        if away_score > home_score and winner != record["away_team"]:
            raise ValueError(
                f"{match_id} winner does not match the score."
            )

        completed[match_id] = record


def add_completed_match_to_state(
    state: dict[str, Any],
    match: dict[str, Any],
    home_team: str,
    away_team: str,
    match_date: str,
    home_score: int,
    away_score: int,
    winner: str,
) -> dict[str, Any]:
    updated_state = copy.deepcopy(state)

    completed = completed_matches_by_id(updated_state)

    if match["id"] in completed:
        raise ValueError("That match has already been completed.")

    record = {
        "match_id": match["id"],
        "round": match["round"],
        "date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": int(home_score),
        "away_score": int(away_score),
        "winner": winner,
    }

    updated_state["completed_matches"].append(record)

    validate_state(updated_state)

    return updated_state


# ============================================================
# Feature Engineering Helpers
# ============================================================

def safe_number(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return float(default)

        return float(value)

    except Exception:
        return float(default)


def first_existing_value(
    row: pd.Series,
    possible_columns: list[str],
    default: float,
) -> float:
    for column in possible_columns:
        if column in row.index and not pd.isna(row[column]):
            return safe_number(row[column], default)

    return float(default)


def get_latest_team_row(
    matches: pd.DataFrame,
    team: str,
) -> pd.Series | None:
    team_matches = matches[
        (matches["home_team"] == team)
        | (matches["away_team"] == team)
    ].sort_values("date")

    if len(team_matches) == 0:
        return None

    return team_matches.iloc[-1]


def get_base_team_profile(
    matches: pd.DataFrame,
    team: str,
) -> dict[str, Any]:
    row = get_latest_team_row(matches, team)

    if row is None:
        return {
            "elo": 1500.0,
            "attack": 1.0,
            "defense": 1.0,
            "streak": 0.0,
            "last_match_date": pd.Timestamp("2026-06-01"),
        }

    if row["home_team"] == team:
        elo = first_existing_value(
            row,
            ["home_elo_before", "home_elo", "home_rating"],
            1500.0,
        )

        attack = first_existing_value(
            row,
            ["home_attack_before", "home_attack", "home_attack_rating"],
            1.0,
        )

        defense = first_existing_value(
            row,
            ["home_defense_before", "home_defense", "home_defense_rating"],
            1.0,
        )

        streak = first_existing_value(
            row,
            ["home_streak", "home_current_streak"],
            0.0,
        )

    else:
        elo = first_existing_value(
            row,
            ["away_elo_before", "away_elo", "away_rating"],
            1500.0,
        )

        attack = first_existing_value(
            row,
            ["away_attack_before", "away_attack", "away_attack_rating"],
            1.0,
        )

        defense = first_existing_value(
            row,
            ["away_defense_before", "away_defense", "away_defense_rating"],
            1.0,
        )

        streak = first_existing_value(
            row,
            ["away_streak", "away_current_streak"],
            0.0,
        )

    return {
        "elo": elo,
        "attack": attack,
        "defense": defense,
        "streak": streak,
        "last_match_date": pd.Timestamp(row["date"]),
    }


def get_all_live_teams(state: dict[str, Any]) -> list[str]:
    teams = set(INITIAL_TEAMS)

    for completed_match in state.get("completed_matches", []):
        teams.add(completed_match["home_team"])
        teams.add(completed_match["away_team"])
        teams.add(completed_match["winner"])

    return sorted(teams)


def build_live_profiles(
    matches: pd.DataFrame,
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    teams = get_all_live_teams(state)

    profiles = {
        team: get_base_team_profile(matches, team)
        for team in teams
    }

    completed_matches = sorted(
        state.get("completed_matches", []),
        key=lambda match: match["date"],
    )

    for completed_match in completed_matches:
        home_team = completed_match["home_team"]
        away_team = completed_match["away_team"]
        home_score = int(completed_match["home_score"])
        away_score = int(completed_match["away_score"])
        match_date = pd.Timestamp(completed_match["date"])

        home_profile = profiles[home_team]
        away_profile = profiles[away_team]

        home_elo = home_profile["elo"]
        away_elo = away_profile["elo"]

        expected_home = 1 / (1 + 10 ** (-(home_elo - away_elo) / 400))

        if home_score > away_score:
            actual_home = 1.0
            home_streak = 1
            away_streak = -1
        elif home_score < away_score:
            actual_home = 0.0
            home_streak = -1
            away_streak = 1
        else:
            actual_home = 0.5
            home_streak = 0
            away_streak = 0

        k_factor = 30

        home_profile["elo"] = home_elo + k_factor * (
            actual_home - expected_home
        )

        away_profile["elo"] = away_elo + k_factor * (
            (1 - actual_home) - (1 - expected_home)
        )

        home_profile["streak"] = home_streak
        away_profile["streak"] = away_streak

        home_profile["last_match_date"] = match_date
        away_profile["last_match_date"] = match_date

    return profiles


def get_recent_form_for_team(
    matches: pd.DataFrame,
    state: dict[str, Any],
    team: str,
    n_matches: int = 5,
) -> dict[str, float]:
    records = []

    historical_matches = matches[
        (matches["home_team"] == team)
        | (matches["away_team"] == team)
    ].copy()

    for _, match in historical_matches.iterrows():
        if pd.isna(match["home_score"]) or pd.isna(match["away_score"]):
            continue

        if match["home_team"] == team:
            goals_for = int(match["home_score"])
            goals_against = int(match["away_score"])
        else:
            goals_for = int(match["away_score"])
            goals_against = int(match["home_score"])

        records.append(
            {
                "date": pd.Timestamp(match["date"]),
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        )

    for completed_match in state.get("completed_matches", []):
        if completed_match["home_team"] == team:
            goals_for = int(completed_match["home_score"])
            goals_against = int(completed_match["away_score"])

        elif completed_match["away_team"] == team:
            goals_for = int(completed_match["away_score"])
            goals_against = int(completed_match["home_score"])

        else:
            continue

        records.append(
            {
                "date": pd.Timestamp(completed_match["date"]),
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        )

    records = sorted(records, key=lambda record: record["date"])
    recent_records = records[-n_matches:]

    if len(recent_records) == 0:
        return {
            "win_rate": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_difference": 0.0,
        }

    wins = 0
    goals_for_total = 0
    goals_against_total = 0

    for record in recent_records:
        goals_for = record["goals_for"]
        goals_against = record["goals_against"]

        goals_for_total += goals_for
        goals_against_total += goals_against

        if goals_for > goals_against:
            wins += 1

    games = len(recent_records)

    return {
        "win_rate": wins / games,
        "goals_for": goals_for_total / games,
        "goals_against": goals_against_total / games,
        "goal_difference": (
            goals_for_total - goals_against_total
        ) / games,
    }


def build_recent_forms(
    matches: pd.DataFrame,
    state: dict[str, Any],
) -> dict[str, dict[str, float]]:
    teams = get_all_live_teams(state)

    return {
        team: get_recent_form_for_team(matches, state, team)
        for team in teams
    }


def get_head_to_head_difference(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> int:
    previous_matches = matches[
        (
            (matches["home_team"] == home_team)
            & (matches["away_team"] == away_team)
        )
        | (
            (matches["home_team"] == away_team)
            & (matches["away_team"] == home_team)
        )
    ]

    home_wins = 0
    away_wins = 0

    for _, match in previous_matches.iterrows():
        if pd.isna(match["home_score"]) or pd.isna(match["away_score"]):
            continue

        if match["home_score"] > match["away_score"]:
            winner = match["home_team"]
        elif match["home_score"] < match["away_score"]:
            winner = match["away_team"]
        else:
            winner = None

        if winner == home_team:
            home_wins += 1
        elif winner == away_team:
            away_wins += 1

    return home_wins - away_wins


def build_live_match_features(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    features: list[str],
    matches: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    recent_forms: dict[str, dict[str, float]],
    neutral: bool = True,
) -> pd.DataFrame:
    home_profile = profiles.get(
        home_team,
        {
            "elo": 1500.0,
            "attack": 1.0,
            "defense": 1.0,
            "streak": 0.0,
            "last_match_date": pd.Timestamp("2026-06-01"),
        },
    )

    away_profile = profiles.get(
        away_team,
        {
            "elo": 1500.0,
            "attack": 1.0,
            "defense": 1.0,
            "streak": 0.0,
            "last_match_date": pd.Timestamp("2026-06-01"),
        },
    )

    home_form = recent_forms.get(
        home_team,
        {
            "win_rate": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_difference": 0.0,
        },
    )

    away_form = recent_forms.get(
        away_team,
        {
            "win_rate": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_difference": 0.0,
        },
    )

    home_rest_days = (
        pd.Timestamp(match_date) - home_profile["last_match_date"]
    ).days

    away_rest_days = (
        pd.Timestamp(match_date) - away_profile["last_match_date"]
    ).days

    row = {
        "elo_difference": home_profile["elo"] - away_profile["elo"],
        "home_advantage": 0 if neutral else 1,
        "recent_win_rate_difference": (
            home_form["win_rate"] - away_form["win_rate"]
        ),
        "recent_goals_for_difference": (
            home_form["goals_for"] - away_form["goals_for"]
        ),
        "recent_goals_against_difference": (
            home_form["goals_against"] - away_form["goals_against"]
        ),
        "recent_goal_difference_difference": (
            home_form["goal_difference"] - away_form["goal_difference"]
        ),
        "rest_days_difference": home_rest_days - away_rest_days,
        "streak_difference": (
            home_profile["streak"] - away_profile["streak"]
        ),
        "is_world_cup": 1,
        "is_qualification": 0,
        "is_friendly": 0,
        "tournament_importance": 4,
        "head_to_head_difference": get_head_to_head_difference(
            matches,
            home_team,
            away_team,
        ),
        "attack_rating_difference": (
            home_profile["attack"] - away_profile["attack"]
        ),
        "defense_rating_difference": (
            home_profile["defense"] - away_profile["defense"]
        ),
    }

    X_live = pd.DataFrame([row])[features]

    X_live = X_live.fillna(
        {
            "elo_difference": 0,
            "home_advantage": 0,
            "recent_win_rate_difference": 0,
            "recent_goals_for_difference": 0,
            "recent_goals_against_difference": 0,
            "recent_goal_difference_difference": 0,
            "rest_days_difference": 0,
            "streak_difference": 0,
            "is_world_cup": 1,
            "is_qualification": 0,
            "is_friendly": 0,
            "tournament_importance": 4,
            "head_to_head_difference": 0,
            "attack_rating_difference": 0,
            "defense_rating_difference": 0,
        }
    )

    return X_live


# ============================================================
# Prediction and Simulation
# ============================================================

def predict_live_match(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    model: Any,
    features: list[str],
    matches: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    recent_forms: dict[str, dict[str, float]],
    prediction_cache: dict[tuple[Any, ...], dict[str, float]],
    state_key: str,
    neutral: bool = True,
) -> dict[str, float]:
    cache_key = (
        state_key,
        home_team,
        away_team,
        str(pd.Timestamp(match_date).date()),
        neutral,
    )

    if cache_key in prediction_cache:
        return prediction_cache[cache_key]

    X_match = build_live_match_features(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        features=features,
        matches=matches,
        profiles=profiles,
        recent_forms=recent_forms,
        neutral=neutral,
    )

    probabilities = model.predict_proba(X_match)[0]

    prediction = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    prediction_cache[cache_key] = prediction

    return prediction


def simulate_live_knockout_match(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    model: Any,
    features: list[str],
    matches: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    recent_forms: dict[str, dict[str, float]],
    prediction_cache: dict[tuple[Any, ...], dict[str, float]],
    state_key: str,
    rng: np.random.Generator,
) -> str:
    probabilities = predict_live_match(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        model=model,
        features=features,
        matches=matches,
        profiles=profiles,
        recent_forms=recent_forms,
        prediction_cache=prediction_cache,
        state_key=state_key,
        neutral=True,
    )

    home_prob = probabilities.get("home_win", 0.0)
    away_prob = probabilities.get("away_win", 0.0)

    total = home_prob + away_prob

    if total <= 0:
        return str(rng.choice([home_team, away_team]))

    home_advances_prob = home_prob / total

    return str(
        rng.choice(
            [home_team, away_team],
            p=[home_advances_prob, 1 - home_advances_prob],
        )
    )


def simulate_live_world_cup_once(
    state: dict[str, Any],
    model: Any,
    features: list[str],
    matches: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    recent_forms: dict[str, dict[str, float]],
    prediction_cache: dict[tuple[Any, ...], dict[str, float]],
    state_key: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    completed = completed_matches_by_id(state)
    simulated_winners = {}

    quarterfinalists = set()
    semifinalists = set()
    finalists = set()
    champion = None

    for match in BRACKET_MATCHES:
        match_id = match["id"]

        home_team, away_team = resolve_match_teams(
            match,
            state,
            simulated_winners,
        )

        if home_team is None or away_team is None:
            continue

        match_date = pd.Timestamp(match["date"])

        if match["round"] == "quarterfinal":
            quarterfinalists.add(home_team)
            quarterfinalists.add(away_team)

        if match_id in completed:
            winner = completed[match_id]["winner"]
        else:
            winner = simulate_live_knockout_match(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                model=model,
                features=features,
                matches=matches,
                profiles=profiles,
                recent_forms=recent_forms,
                prediction_cache=prediction_cache,
                state_key=state_key,
                rng=rng,
            )

        simulated_winners[match_id] = winner

        if match["round"] == "quarterfinal":
            semifinalists.add(winner)

        elif match["round"] == "semifinal":
            finalists.add(winner)

        elif match["round"] == "final":
            champion = winner

    return {
        "quarterfinalists": list(quarterfinalists),
        "semifinalists": list(semifinalists),
        "finalists": list(finalists),
        "champion": champion,
    }


def run_live_simulations(
    state: dict[str, Any],
    model: Any,
    features: list[str],
    matches: pd.DataFrame,
    num_simulations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    profiles = build_live_profiles(matches, state)
    recent_forms = build_recent_forms(matches, state)
    prediction_cache = {}
    state_key = state_signature(state)

    teams = set(INITIAL_TEAMS)

    for completed_match in state.get("completed_matches", []):
        teams.add(completed_match["home_team"])
        teams.add(completed_match["away_team"])
        teams.add(completed_match["winner"])

    live_results = {}

    for team in sorted(teams):
        live_results[team] = {
            "quarterfinal": 0,
            "semifinal": 0,
            "final": 0,
            "champion": 0,
        }

    for _ in range(num_simulations):
        simulation = simulate_live_world_cup_once(
            state=state,
            model=model,
            features=features,
            matches=matches,
            profiles=profiles,
            recent_forms=recent_forms,
            prediction_cache=prediction_cache,
            state_key=state_key,
            rng=rng,
        )

        for team in simulation["quarterfinalists"]:
            live_results[team]["quarterfinal"] += 1

        for team in simulation["semifinalists"]:
            live_results[team]["semifinal"] += 1

        for team in simulation["finalists"]:
            live_results[team]["final"] += 1

        champion = simulation["champion"]

        if champion is not None:
            live_results[champion]["champion"] += 1

    results = pd.DataFrame.from_dict(live_results, orient="index")
    results.index.name = "team"

    results = results / num_simulations
    results = results.sort_values("champion", ascending=False)

    return results


# ============================================================
# UI Helpers
# ============================================================

def probability_to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def display_top_three(results: pd.DataFrame) -> None:
    top_three = results.head(3)

    columns = st.columns(3)

    for column, (team, row) in zip(columns, top_three.iterrows()):
        column.metric(
            label=team,
            value=probability_to_percent(row["champion"]),
            delta="Champion odds",
        )


def make_bar_chart(
    results: pd.DataFrame,
    column_name: str,
    title: str,
    top_n: int = 10,
):
    chart_data = (
        results[column_name]
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    chart_data[column_name] = chart_data[column_name] * 100

    fig = px.bar(
        chart_data,
        x="team",
        y=column_name,
        title=title,
        labels={
            "team": "Team",
            column_name: "Probability (%)",
        },
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        height=450,
    )

    return fig


def make_history_chart(history: pd.DataFrame):
    if history.empty:
        return None

    required_columns = {
        "snapshot_label",
        "team",
        "champion",
    }

    if not required_columns.issubset(set(history.columns)):
        return None

    chart_data = history.copy()
    chart_data["champion_pct"] = chart_data["champion"] * 100

    top_teams = (
        chart_data.sort_values("champion", ascending=False)
        .groupby("snapshot_label")
        .head(3)["team"]
        .unique()
        .tolist()
    )

    chart_data = chart_data[chart_data["team"].isin(top_teams)]

    fig = px.line(
        chart_data,
        x="snapshot_label",
        y="champion_pct",
        color="team",
        markers=True,
        title="Champion Probability Over Time",
        labels={
            "snapshot_label": "Prediction Snapshot",
            "champion_pct": "Champion Probability (%)",
            "team": "Team",
        },
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        height=450,
    )

    return fig


def display_completed_matches(state: dict[str, Any]) -> None:
    completed = state.get("completed_matches", [])

    if len(completed) == 0:
        st.info("No completed match results are locked in yet.")
        return

    completed_df = pd.DataFrame(completed)

    display_columns = [
        "date",
        "round",
        "home_team",
        "home_score",
        "away_score",
        "away_team",
        "winner",
    ]

    st.dataframe(
        completed_df[display_columns],
        width="stretch",
        hide_index=True,
    )


def make_current_snapshot(
    results: pd.DataFrame,
    snapshot_label: str,
) -> pd.DataFrame:
    snapshot = results.reset_index().rename(columns={"index": "team"})

    if "team" not in snapshot.columns:
        snapshot = snapshot.rename(columns={snapshot.columns[0]: "team"})

    snapshot.insert(0, "snapshot_label", snapshot_label)

    return snapshot


def initialize_scenario_state(official_state: dict[str, Any]) -> None:
    official_key = state_signature(official_state)

    if "official_state_key" not in st.session_state:
        st.session_state["official_state_key"] = official_key

    if "scenario_state" not in st.session_state:
        st.session_state["scenario_state"] = copy.deepcopy(official_state)

    if "scenario_history" not in st.session_state:
        st.session_state["scenario_history"] = []

    if st.session_state["official_state_key"] != official_key:
        st.session_state["official_state_key"] = official_key
        st.session_state["scenario_state"] = copy.deepcopy(official_state)
        st.session_state["scenario_history"] = []


def reset_scenario_state(official_state: dict[str, Any]) -> None:
    st.session_state["scenario_state"] = copy.deepcopy(official_state)
    st.session_state["scenario_history"] = []


# ============================================================
# Main App
# ============================================================

st.title("⚽ FIFA World Cup ML Predictor")
st.caption(
    "A live tournament dashboard powered by a trained match prediction model, "
    "live feature updates, and Monte Carlo simulation."
)

try:
    model, features = load_model_and_features()
    matches = load_matches()
    official_state = load_official_state()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except ValueError as error:
    st.error(f"Official results file has an error: {error}")
    st.stop()


initialize_scenario_state(official_state)

with st.sidebar:
    st.header("Simulation Settings")

    num_simulations = st.slider(
        "Number of simulations",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500,
    )

    seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
    )

    st.divider()

    st.write("Official results are read-only on the public app.")

    if st.button("Reset My Scenario"):
        reset_scenario_state(official_state)
        st.success("Your temporary scenario was reset.")
        st.rerun()


with st.spinner("Running official tournament simulations..."):
    official_results = run_live_simulations(
        state=official_state,
        model=model,
        features=features,
        matches=matches,
        num_simulations=num_simulations,
        seed=int(seed),
    )


dashboard_tab, scenario_tab, history_tab, match_tab, about_tab = st.tabs(
    [
        "Dashboard",
        "Scenario Lab",
        "Past Predictions",
        "Match Probabilities",
        "About the Model",
    ]
)


# ============================================================
# Dashboard Tab
# ============================================================

with dashboard_tab:
    st.subheader("Official Current Projection")

    projected_champion = official_results.index[0]
    projected_champion_probability = official_results.iloc[0]["champion"]

    st.metric(
        label="Projected Champion",
        value=projected_champion,
        delta=probability_to_percent(projected_champion_probability),
    )

    st.markdown("### Top 3 Champion Odds")
    display_top_three(official_results)

    st.markdown("### Locked Official Results")
    display_completed_matches(official_state)

    st.markdown("### Champion Probabilities")
    champion_fig = make_bar_chart(
        official_results,
        "champion",
        "Official Champion Probabilities",
    )
    st.plotly_chart(champion_fig, width="stretch")

    st.markdown("### Final Appearance Probabilities")
    final_fig = make_bar_chart(
        official_results,
        "final",
        "Official Final Appearance Probabilities",
    )
    st.plotly_chart(final_fig, width="stretch")

    st.markdown("### Full Official Probability Table")
    st.dataframe(
        (official_results * 100).round(2),
        width="stretch",
    )


# ============================================================
# Scenario Lab Tab
# ============================================================

with scenario_tab:
    st.subheader("Scenario Lab")

    st.write(
        "Try hypothetical match results and see how the model changes. "
        "These updates are temporary and only affect your browser session. "
        "Refreshing the page resets back to the official dashboard state."
    )

    scenario_state = st.session_state["scenario_state"]

    with st.spinner("Running scenario simulations..."):
        scenario_results = run_live_simulations(
            state=scenario_state,
            model=model,
            features=features,
            matches=matches,
            num_simulations=num_simulations,
            seed=int(seed),
        )

    st.markdown("### Scenario Projection")

    scenario_projected_champion = scenario_results.index[0]
    scenario_projected_champion_probability = scenario_results.iloc[0][
        "champion"
    ]

    st.metric(
        label="Scenario Projected Champion",
        value=scenario_projected_champion,
        delta=probability_to_percent(scenario_projected_champion_probability),
    )

    display_top_three(scenario_results)

    st.markdown("### Scenario Results Entered")
    display_completed_matches(scenario_state)

    st.divider()

    available_matches = get_available_matches(scenario_state)

    if len(available_matches) == 0:
        st.success("All matches have been completed in this scenario.")

    else:
        option_labels = [
            format_match_label(match, home_team, away_team)
            for match, home_team, away_team in available_matches
        ]

        selected_label = st.selectbox(
            "Select a match to update temporarily",
            option_labels,
        )

        selected_index = option_labels.index(selected_label)
        selected_match, home_team, away_team = available_matches[
            selected_index
        ]

        with st.form("scenario_result_form"):
            match_date = st.date_input(
                "Match date",
                value=pd.Timestamp(selected_match["date"]).date(),
            )

            left_column, right_column = st.columns(2)

            with left_column:
                home_score = st.number_input(
                    f"{home_team} score",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                )

            with right_column:
                away_score = st.number_input(
                    f"{away_team} score",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                )

            if home_score > away_score:
                winner = home_team
                st.info(f"Winner: {winner}")

            elif away_score > home_score:
                winner = away_team
                st.info(f"Winner: {winner}")

            else:
                winner = st.selectbox(
                    "Scores are tied. Who advances?",
                    [home_team, away_team],
                )

            submitted = st.form_submit_button("Apply Temporary Scenario Result")

        if submitted:
            try:
                updated_scenario_state = add_completed_match_to_state(
                    state=scenario_state,
                    match=selected_match,
                    home_team=home_team,
                    away_team=away_team,
                    match_date=str(match_date),
                    home_score=int(home_score),
                    away_score=int(away_score),
                    winner=winner,
                )

                st.session_state["scenario_state"] = updated_scenario_state

                label = (
                    f"After {home_team} "
                    f"{int(home_score)}-{int(away_score)} {away_team}"
                )

                st.session_state["scenario_history"].append(label)

                st.success(
                    "Temporary scenario applied. "
                    "This did not save to the official dashboard."
                )

                st.rerun()

            except ValueError as error:
                st.error(str(error))

    st.markdown("### Scenario Champion Probabilities")
    scenario_champion_fig = make_bar_chart(
        scenario_results,
        "champion",
        "Scenario Champion Probabilities",
    )
    st.plotly_chart(scenario_champion_fig, width="stretch")

    st.markdown("### Scenario Full Probability Table")
    st.dataframe(
        (scenario_results * 100).round(2),
        width="stretch",
    )


# ============================================================
# Past Predictions Tab
# ============================================================

with history_tab:
    st.subheader("Past Official Prediction Snapshots")

    history = load_prediction_history()

    if history.empty:
        st.info(
            "No committed prediction history file found yet. "
            "The current official snapshot is shown below."
        )

        current_snapshot = make_current_snapshot(
            official_results,
            "Current official projection",
        )

        display_snapshot = current_snapshot.copy()

        for column in ["quarterfinal", "semifinal", "final", "champion"]:
            display_snapshot[column] = (
                display_snapshot[column] * 100
            ).round(2)

        st.dataframe(
            display_snapshot,
            width="stretch",
            hide_index=True,
        )

    else:
        history_chart = make_history_chart(history)

        if history_chart is not None:
            st.plotly_chart(history_chart, width="stretch")

        display_history = history.copy()

        for column in ["quarterfinal", "semifinal", "final", "champion"]:
            if column in display_history.columns:
                display_history[column] = (
                    display_history[column] * 100
                ).round(2)

        st.dataframe(
            display_history,
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Your Temporary Scenario Path")

    if len(st.session_state["scenario_history"]) == 0:
        st.info("You have not applied any temporary scenario results yet.")
    else:
        for index, label in enumerate(st.session_state["scenario_history"], 1):
            st.write(f"{index}. {label}")


# ============================================================
# Match Probabilities Tab
# ============================================================

with match_tab:
    st.subheader("Current Official Match Probability Explorer")

    available_matches = get_available_matches(official_state)

    if len(available_matches) == 0:
        st.info("No unresolved official matches are currently available.")

    else:
        option_labels = [
            format_match_label(match, home_team, away_team)
            for match, home_team, away_team in available_matches
        ]

        selected_label = st.selectbox(
            "Select official unresolved match",
            option_labels,
            key="official_probability_selector",
        )

        selected_index = option_labels.index(selected_label)
        selected_match, home_team, away_team = available_matches[
            selected_index
        ]

        profiles = build_live_profiles(matches, official_state)
        recent_forms = build_recent_forms(matches, official_state)
        prediction_cache = {}

        probabilities = predict_live_match(
            home_team=home_team,
            away_team=away_team,
            match_date=pd.Timestamp(selected_match["date"]),
            model=model,
            features=features,
            matches=matches,
            profiles=profiles,
            recent_forms=recent_forms,
            prediction_cache=prediction_cache,
            state_key=state_signature(official_state),
            neutral=True,
        )

        home_win_probability = probabilities.get("home_win", 0.0)
        draw_probability = probabilities.get("draw", 0.0)
        away_win_probability = probabilities.get("away_win", 0.0)

        knockout_total = home_win_probability + away_win_probability

        if knockout_total > 0:
            home_advances_probability = (
                home_win_probability / knockout_total
            )
            away_advances_probability = (
                away_win_probability / knockout_total
            )
        else:
            home_advances_probability = 0.5
            away_advances_probability = 0.5

        probability_table = pd.DataFrame(
            [
                {
                    "Outcome": f"{home_team} wins in model outcome",
                    "Probability": home_win_probability,
                },
                {
                    "Outcome": "Draw in model outcome",
                    "Probability": draw_probability,
                },
                {
                    "Outcome": f"{away_team} wins in model outcome",
                    "Probability": away_win_probability,
                },
                {
                    "Outcome": f"{home_team} advances after draw removed",
                    "Probability": home_advances_probability,
                },
                {
                    "Outcome": f"{away_team} advances after draw removed",
                    "Probability": away_advances_probability,
                },
            ]
        )

        probability_table["Probability"] = (
            probability_table["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_table,
            width="stretch",
            hide_index=True,
        )

        fig = px.bar(
            probability_table,
            x="Outcome",
            y="Probability",
            title=f"{home_team} vs {away_team} Probability Breakdown",
            labels={
                "Outcome": "Outcome",
                "Probability": "Probability (%)",
            },
        )

        fig.update_layout(
            xaxis_tickangle=-30,
            height=450,
        )

        st.plotly_chart(fig, width="stretch")


# ============================================================
# About Tab
# ============================================================

with about_tab:
    st.subheader("About This Project")

    st.markdown(
        """
        This dashboard turns the FIFA World Cup machine learning project into
        an interactive portfolio app.

        The trained model predicts three match outcomes:

        - home win
        - draw
        - away win

        The tournament simulator then removes draw outcomes for knockout
        matches and normalizes the win probabilities so that one team advances.
        """
    )

    st.markdown("### Final Model Selected")

    st.write("Gradient Boosting was selected as the best model from Notebook 09.")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": "Gradient Boosting",
                    "Accuracy": 0.603610,
                    "Log Loss": 0.872078,
                },
                {
                    "Model": "Random Forest",
                    "Accuracy": 0.580559,
                    "Log Loss": 0.897091,
                },
                {
                    "Model": "Logistic Regression",
                    "Accuracy": 0.581168,
                    "Log Loss": 0.899839,
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Features Used")

    st.code("\n".join(features))

    st.markdown("### How Updates Work")

    st.markdown(
        """
        Official real-world results are stored in:

        `data/app/official_results.json`

        That file is committed to the repository. Public users cannot edit it
        from the website.

        The Scenario Lab uses Streamlit session state, so visitors can enter
        hypothetical scores and watch the probabilities change, but those
        changes are temporary. Refreshing the page resets the app back to the
        official committed state.
        """
    )

    st.markdown("### Limitations")

    st.markdown(
        """
        - The model does not include injuries, lineups, betting markets, or player-level data.
        - The app updates live features and bracket state, but it does not retrain the model after every match.
        - Knockout draws are handled by normalizing win probabilities.
        - The dashboard is designed as a portfolio demo, not a production betting or forecasting system.
        """
    )

    st.info(
        "Portfolio summary: this project trains a football match prediction "
        "model, selects the best classifier, and deploys an interactive "
        "dashboard where users can test match-result scenarios and see how "
        "World Cup probabilities change."
    )