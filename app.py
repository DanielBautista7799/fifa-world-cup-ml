from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Project Paths and Constants
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "best_match_prediction_model.pkl"
FEATURES_PATH = ROOT / "models" / "model_features.pkl"

MATCH_DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "matches_with_advanced_features.csv",
    ROOT / "data" / "processed" / "advanced_match_features.csv",
]

OFFICIAL_RESULTS_PATH = ROOT / "data" / "app" / "official_results.json"
TOURNAMENT_SUMMARY_PATH = ROOT / "data" / "app" / "tournament_summary.json"
PREDICTION_HISTORY_PATH = ROOT / "data" / "app" / "prediction_history.csv"
SNAPSHOT_SUMMARY_PATH = (
    ROOT / "data" / "app" / "prediction_snapshot_summary.csv"
)
FINAL_MODEL_REPORT_PATH = ROOT / "data" / "app" / "final_model_report.json"

# The history starts after the two July 4 Round of 16 matches that produced
# Morocco vs France. Any processed rows on or after this date are excluded
# from the base feature snapshot so replaying old checkpoints cannot leak
# information from later tournament results.
TRACKING_START_DATE = pd.Timestamp("2026-07-05")
ACTUAL_CHAMPION = "Spain"

DEFAULT_STATE: dict[str, list[dict[str, Any]]] = {
    "completed_matches": []
}

MODEL_RESULTS = pd.DataFrame(
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
)


# ============================================================
# Bracket Definition
# ============================================================

BRACKET_MATCHES: list[dict[str, Any]] = [
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
# File Loading Helpers
# ============================================================


def find_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_model_and_features(
    model_path: Path = MODEL_PATH,
    features_path: Path = FEATURES_PATH,
) -> tuple[Any, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing model file: models/best_match_prediction_model.pkl"
        )

    if not features_path.exists():
        raise FileNotFoundError(
            "Missing feature file: models/model_features.pkl"
        )

    model = joblib.load(model_path)
    features = list(joblib.load(features_path))

    return model, features


def load_matches(
    candidate_paths: Iterable[Path] = MATCH_DATA_CANDIDATES,
) -> pd.DataFrame:
    match_path = find_existing_path(candidate_paths)

    if match_path is None:
        raise FileNotFoundError(
            "Missing processed match data. Expected either "
            "data/processed/matches_with_advanced_features.csv or "
            "data/processed/advanced_match_features.csv."
        )

    matches = pd.read_csv(match_path)

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing_columns = required_columns - set(matches.columns)

    if missing_columns:
        raise ValueError(
            "Processed match data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    return matches


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path.name} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")

    return value


def get_clean_default_state() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_STATE)


def load_official_state(
    path: Path = OFFICIAL_RESULTS_PATH,
) -> dict[str, Any]:
    if not path.exists():
        return get_clean_default_state()

    state = load_json(path)
    state.setdefault("completed_matches", [])
    validate_state(state)
    return state


# ============================================================
# Bracket and State Helpers
# ============================================================


def get_match_by_id(match_id: str) -> dict[str, Any]:
    for match in BRACKET_MATCHES:
        if match["id"] == match_id:
            return match

    raise ValueError(f"Unknown match_id: {match_id}")


def completed_matches_by_id(
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(match["match_id"]): match
        for match in state.get("completed_matches", [])
    }


def state_signature(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


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
        return str(completed[source_id]["winner"])

    return None


def resolve_match_teams(
    match: dict[str, Any],
    state: dict[str, Any],
    simulated_winners: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    simulated_winners = simulated_winners or {}
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


def validate_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ValueError("The official results file must be a JSON object.")

    records = state.get("completed_matches")
    if not isinstance(records, list):
        raise ValueError(
            'The official results file must contain a "completed_matches" list.'
        )

    required_fields = {
        "match_id",
        "round",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "winner",
    }

    seen_match_ids: set[str] = set()
    validated_records: list[dict[str, Any]] = []
    previous_date: pd.Timestamp | None = None

    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Completed match #{record_number} must be a JSON object."
            )

        missing_fields = required_fields - set(record)
        if missing_fields:
            raise ValueError(
                f"Completed match #{record_number} is missing fields: "
                f"{', '.join(sorted(missing_fields))}."
            )

        match_id = str(record["match_id"])

        if match_id in seen_match_ids:
            raise ValueError(f'Duplicate result for match_id "{match_id}".')

        definition = get_match_by_id(match_id)
        prior_state = {
            "completed_matches": copy.deepcopy(validated_records)
        }
        expected_home, expected_away = resolve_match_teams(
            definition,
            prior_state,
        )

        if expected_home is None or expected_away is None:
            raise ValueError(
                f'Match "{match_id}" cannot be recorded yet because an '
                "earlier result needed to resolve its teams is missing or "
                "appears later in the file."
            )

        if record["home_team"] != expected_home:
            raise ValueError(
                f'Match "{match_id}" has the wrong home team. '
                f'Expected "{expected_home}", got "{record["home_team"]}".'
            )

        if record["away_team"] != expected_away:
            raise ValueError(
                f'Match "{match_id}" has the wrong away team. '
                f'Expected "{expected_away}", got "{record["away_team"]}".'
            )

        if record["round"] != definition["round"]:
            raise ValueError(
                f'Match "{match_id}" has the wrong round. '
                f'Expected "{definition["round"]}", got '
                f'"{record["round"]}".'
            )

        try:
            match_date = pd.Timestamp(record["date"]).normalize()
        except Exception as error:
            raise ValueError(
                f'Match "{match_id}" has an invalid date: '
                f'{record["date"]}.'
            ) from error

        expected_date = pd.Timestamp(definition["date"]).normalize()
        if match_date != expected_date:
            raise ValueError(
                f'Match "{match_id}" has the wrong date. '
                f'Expected {expected_date.date()}, got {match_date.date()}.'
            )

        if previous_date is not None and match_date < previous_date:
            raise ValueError(
                "Official results must be listed in chronological order."
            )

        try:
            home_score = int(record["home_score"])
            away_score = int(record["away_score"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Match "{match_id}" scores must be whole numbers.'
            ) from error

        if not 0 <= home_score <= 20 or not 0 <= away_score <= 20:
            raise ValueError(
                f'Match "{match_id}" scores must be between 0 and 20.'
            )

        winner = str(record["winner"])
        valid_winners = {expected_home, expected_away}

        if winner not in valid_winners:
            raise ValueError(
                f'Match "{match_id}" winner must be either '
                f'"{expected_home}" or "{expected_away}".'
            )

        if home_score > away_score and winner != expected_home:
            raise ValueError(
                f'Match "{match_id}" winner does not match the score.'
            )

        if away_score > home_score and winner != expected_away:
            raise ValueError(
                f'Match "{match_id}" winner does not match the score.'
            )

        validated_records.append(
            {
                "match_id": match_id,
                "round": definition["round"],
                "date": str(match_date.date()),
                "home_team": expected_home,
                "away_team": expected_away,
                "home_score": home_score,
                "away_score": away_score,
                "winner": winner,
            }
        )
        seen_match_ids.add(match_id)
        previous_date = match_date


def get_state_at_checkpoint(
    official_state: dict[str, Any],
    checkpoint_index: int,
) -> dict[str, Any]:
    records = official_state.get("completed_matches", [])

    if not 0 <= checkpoint_index <= len(records):
        raise ValueError(
            f"checkpoint_index must be between 0 and {len(records)}."
        )

    return {
        "completed_matches": copy.deepcopy(records[:checkpoint_index])
    }


def get_available_matches(
    state: dict[str, Any],
) -> list[tuple[dict[str, Any], str, str]]:
    completed = completed_matches_by_id(state)
    available: list[tuple[dict[str, Any], str, str]] = []

    for match in BRACKET_MATCHES:
        if match["id"] in completed:
            continue

        home_team, away_team = resolve_match_teams(match, state)
        if home_team is not None and away_team is not None:
            available.append((match, home_team, away_team))

    return available


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
    if match["id"] in completed_matches_by_id(state):
        raise ValueError("That match is already completed in this scenario.")

    updated_state = copy.deepcopy(state)
    updated_state.setdefault("completed_matches", [])
    updated_state["completed_matches"].append(
        {
            "match_id": match["id"],
            "round": match["round"],
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "winner": winner,
        }
    )

    validate_state(updated_state)
    return updated_state


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


# ============================================================
# Time-Safe Feature Engineering
# ============================================================


def get_base_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Return only rows known before the first archived checkpoint."""
    return matches[matches["date"] < TRACKING_START_DATE].copy()


def safe_number(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
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

    if team_matches.empty:
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
            [
                "home_attack_before",
                "home_attack",
                "home_attack_rating",
            ],
            1.0,
        )
        defense = first_existing_value(
            row,
            [
                "home_defense_before",
                "home_defense",
                "home_defense_rating",
            ],
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
            [
                "away_attack_before",
                "away_attack",
                "away_attack_rating",
            ],
            1.0,
        )
        defense = first_existing_value(
            row,
            [
                "away_defense_before",
                "away_defense",
                "away_defense_rating",
            ],
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
        teams.update(
            {
                completed_match["home_team"],
                completed_match["away_team"],
                completed_match["winner"],
            }
        )

    return sorted(teams)


def build_live_profiles(
    matches: pd.DataFrame,
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    base_matches = get_base_matches(matches)
    profiles = {
        team: get_base_team_profile(base_matches, team)
        for team in get_all_live_teams(state)
    }

    completed_matches = sorted(
        state.get("completed_matches", []),
        key=lambda match: (
            pd.Timestamp(match["date"]),
            BRACKET_MATCHES.index(get_match_by_id(match["match_id"])),
        ),
    )

    for completed_match in completed_matches:
        home_team = str(completed_match["home_team"])
        away_team = str(completed_match["away_team"])
        home_score = int(completed_match["home_score"])
        away_score = int(completed_match["away_score"])
        match_date = pd.Timestamp(completed_match["date"])

        home_profile = profiles[home_team]
        away_profile = profiles[away_team]

        home_elo = float(home_profile["elo"])
        away_elo = float(away_profile["elo"])

        expected_home = 1 / (1 + 10 ** (-(home_elo - away_elo) / 400))

        if home_score > away_score:
            actual_home = 1.0
            home_streak = 1.0
            away_streak = -1.0
        elif home_score < away_score:
            actual_home = 0.0
            home_streak = -1.0
            away_streak = 1.0
        else:
            # A penalty shootout winner does not change the tied match score.
            actual_home = 0.5
            home_streak = 0.0
            away_streak = 0.0

        k_factor = 30.0
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
    base_matches = get_base_matches(matches)
    records: list[dict[str, Any]] = []

    historical_matches = base_matches[
        (base_matches["home_team"] == team)
        | (base_matches["away_team"] == team)
    ]

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

    recent_records = sorted(
        records,
        key=lambda record: record["date"],
    )[-n_matches:]

    if not recent_records:
        return {
            "win_rate": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_difference": 0.0,
        }

    games = len(recent_records)
    wins = sum(
        1
        for record in recent_records
        if record["goals_for"] > record["goals_against"]
    )
    goals_for_total = sum(
        record["goals_for"] for record in recent_records
    )
    goals_against_total = sum(
        record["goals_against"] for record in recent_records
    )

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
    return {
        team: get_recent_form_for_team(matches, state, team)
        for team in get_all_live_teams(state)
    }


def get_head_to_head_difference(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> int:
    base_matches = get_base_matches(matches)
    previous_matches = base_matches[
        (
            (base_matches["home_team"] == home_team)
            & (base_matches["away_team"] == away_team)
        )
        | (
            (base_matches["home_team"] == away_team)
            & (base_matches["away_team"] == home_team)
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
    default_profile = {
        "elo": 1500.0,
        "attack": 1.0,
        "defense": 1.0,
        "streak": 0.0,
        "last_match_date": pd.Timestamp("2026-06-01"),
    }
    default_form = {
        "win_rate": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "goal_difference": 0.0,
    }

    home_profile = profiles.get(home_team, default_profile)
    away_profile = profiles.get(away_team, default_profile)
    home_form = recent_forms.get(home_team, default_form)
    away_form = recent_forms.get(away_team, default_form)

    home_rest_days = (
        pd.Timestamp(match_date)
        - pd.Timestamp(home_profile["last_match_date"])
    ).days
    away_rest_days = (
        pd.Timestamp(match_date)
        - pd.Timestamp(away_profile["last_match_date"])
    ).days

    row = {
        "elo_difference": (
            float(home_profile["elo"])
            - float(away_profile["elo"])
        ),
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
            home_form["goal_difference"]
            - away_form["goal_difference"]
        ),
        "rest_days_difference": home_rest_days - away_rest_days,
        "streak_difference": (
            float(home_profile["streak"])
            - float(away_profile["streak"])
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
            float(home_profile["attack"])
            - float(away_profile["attack"])
        ),
        "defense_rating_difference": (
            float(home_profile["defense"])
            - float(away_profile["defense"])
        ),
    }

    unsupported_features = [
        feature for feature in features if feature not in row
    ]
    if unsupported_features:
        raise ValueError(
            "The saved model expects unsupported features: "
            f"{unsupported_features}"
        )

    return pd.DataFrame([row], columns=features).fillna(0)


# ============================================================
# Prediction and Monte Carlo Simulation
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
        for class_name, probability in zip(
            model.classes_,
            probabilities,
        )
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

    home_probability = probabilities.get("home_win", 0.0)
    away_probability = probabilities.get("away_win", 0.0)
    knockout_total = home_probability + away_probability

    if knockout_total <= 0:
        home_advances_probability = 0.5
    else:
        home_advances_probability = home_probability / knockout_total

    return str(
        rng.choice(
            [home_team, away_team],
            p=[
                home_advances_probability,
                1 - home_advances_probability,
            ],
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
    simulated_winners: dict[str, str] = {}

    quarterfinalists: set[str] = set()
    semifinalists: set[str] = set()
    finalists: set[str] = set()
    champion: str | None = None

    for match in BRACKET_MATCHES:
        match_id = match["id"]
        home_team, away_team = resolve_match_teams(
            match,
            state,
            simulated_winners,
        )

        if home_team is None or away_team is None:
            continue

        if match["round"] == "quarterfinal":
            quarterfinalists.update({home_team, away_team})

        if match_id in completed:
            winner = str(completed[match_id]["winner"])
        else:
            winner = simulate_live_knockout_match(
                home_team=home_team,
                away_team=away_team,
                match_date=pd.Timestamp(match["date"]),
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
    num_simulations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    validate_state(state)

    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive.")

    rng = np.random.default_rng(seed)
    profiles = build_live_profiles(matches, state)
    recent_forms = build_recent_forms(matches, state)
    prediction_cache: dict[tuple[Any, ...], dict[str, float]] = {}
    state_key = state_signature(state)

    results = {
        team: {
            "quarterfinal": 0,
            "semifinal": 0,
            "final": 0,
            "champion": 0,
        }
        for team in INITIAL_TEAMS
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
            results[team]["quarterfinal"] += 1

        for team in simulation["semifinalists"]:
            results[team]["semifinal"] += 1

        for team in simulation["finalists"]:
            results[team]["final"] += 1

        champion = simulation["champion"]
        if champion is not None:
            results[champion]["champion"] += 1

    result_frame = pd.DataFrame.from_dict(results, orient="index")
    result_frame.index.name = "team"

    return (result_frame / num_simulations).sort_values(
        "champion",
        ascending=False,
    )


# ============================================================
# Permanent Prediction History Builder
# ============================================================


def format_result_summary(record: dict[str, Any]) -> str:
    home_team = str(record["home_team"])
    away_team = str(record["away_team"])
    home_score = int(record["home_score"])
    away_score = int(record["away_score"])
    winner = str(record["winner"])

    if home_score == away_score:
        return (
            f"{winner} advanced after {home_team} "
            f"{home_score}-{away_score} {away_team}"
        )

    return f"{home_team} {home_score}-{away_score} {away_team}"


def checkpoint_label(
    checkpoint_index: int,
    records: list[dict[str, Any]],
) -> str:
    if checkpoint_index == 0:
        return "Initial projection before the July 5 updates"

    record = records[checkpoint_index - 1]
    return f"After {format_result_summary(record)}"


def checkpoint_short_label(
    checkpoint_index: int,
    records: list[dict[str, Any]],
) -> str:
    if checkpoint_index == 0:
        return "Start"

    record = records[checkpoint_index - 1]
    timestamp = pd.Timestamp(record["date"])
    date_text = f"{timestamp.strftime('%b')} {timestamp.day}"
    winner = str(record["winner"])
    return f"{date_text}: {winner}"


def teams_remaining_count(state: dict[str, Any]) -> int:
    eliminated: set[str] = set()

    for match in state.get("completed_matches", []):
        loser = (
            match["away_team"]
            if match["winner"] == match["home_team"]
            else match["home_team"]
        )
        eliminated.add(str(loser))

    return len(set(INITIAL_TEAMS) - eliminated)


def build_prediction_history(
    official_state: dict[str, Any],
    model: Any,
    features: list[str],
    matches: pd.DataFrame,
    num_simulations: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_state(official_state)
    records = official_state["completed_matches"]

    history_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for checkpoint_index in range(len(records) + 1):
        checkpoint_state = get_state_at_checkpoint(
            official_state,
            checkpoint_index,
        )
        label = checkpoint_label(checkpoint_index, records)
        short_label = checkpoint_short_label(checkpoint_index, records)
        trigger = records[checkpoint_index - 1] if checkpoint_index else None

        checkpoint_results = run_live_simulations(
            state=checkpoint_state,
            model=model,
            features=features,
            matches=matches,
            num_simulations=num_simulations,
            seed=seed,
        )

        ranked = checkpoint_results.reset_index().copy()
        ranked["champion_rank"] = (
            ranked["champion"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        top_row = ranked.iloc[0]
        champion_row = ranked[
            ranked["team"] == ACTUAL_CHAMPION
        ].iloc[0]

        snapshot_id = (
            "initial"
            if trigger is None
            else f"after_{trigger['match_id']}"
        )
        snapshot_date = (
            str(TRACKING_START_DATE.date())
            if trigger is None
            else str(trigger["date"])
        )
        trigger_match_id = "" if trigger is None else str(trigger["match_id"])
        trigger_result = "" if trigger is None else format_result_summary(trigger)

        summary_row = {
            "checkpoint_index": checkpoint_index,
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date,
            "snapshot_label": label,
            "snapshot_short_label": short_label,
            "trigger_match_id": trigger_match_id,
            "trigger_result": trigger_result,
            "teams_remaining": teams_remaining_count(checkpoint_state),
            "top_projected_champion": str(top_row["team"]),
            "top_champion_probability": float(top_row["champion"]),
            "actual_champion": ACTUAL_CHAMPION,
            "actual_champion_probability": float(
                champion_row["champion"]
            ),
            "actual_champion_rank": int(
                champion_row["champion_rank"]
            ),
        }
        summary_rows.append(summary_row)

        for _, row in ranked.iterrows():
            history_rows.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "snapshot_id": snapshot_id,
                    "snapshot_date": snapshot_date,
                    "snapshot_label": label,
                    "snapshot_short_label": short_label,
                    "trigger_match_id": trigger_match_id,
                    "trigger_result": trigger_result,
                    "team": str(row["team"]),
                    "quarterfinal": float(row["quarterfinal"]),
                    "semifinal": float(row["semifinal"]),
                    "final": float(row["final"]),
                    "champion": float(row["champion"]),
                    "champion_rank": int(row["champion_rank"]),
                }
            )

    history = pd.DataFrame(history_rows)
    snapshot_summary = pd.DataFrame(summary_rows)

    pre_final_checkpoint = max(len(records) - 1, 0)
    pre_final = history[
        history["checkpoint_index"] == pre_final_checkpoint
    ].sort_values("champion", ascending=False)

    initial_summary = snapshot_summary.iloc[0]
    before_final_summary = snapshot_summary.iloc[pre_final_checkpoint]
    champion_before_final = pre_final[
        pre_final["team"] == ACTUAL_CHAMPION
    ].iloc[0]

    final_report = {
        "actual_champion": ACTUAL_CHAMPION,
        "history_method": (
            "Replayed every chronological prefix of official_results.json "
            "using only processed match rows dated before 2026-07-05, then "
            "applied completed tournament matches one at a time."
        ),
        "base_data_cutoff_exclusive": str(TRACKING_START_DATE.date()),
        "total_checkpoints": int(len(snapshot_summary)),
        "simulation_count_per_checkpoint": int(num_simulations),
        "random_seed": int(seed),
        "initial_projected_champion": str(
            initial_summary["top_projected_champion"]
        ),
        "initial_projected_champion_probability": float(
            initial_summary["top_champion_probability"]
        ),
        "initial_actual_champion_probability": float(
            initial_summary["actual_champion_probability"]
        ),
        "initial_actual_champion_rank": int(
            initial_summary["actual_champion_rank"]
        ),
        "pre_final_projected_champion": str(
            before_final_summary["top_projected_champion"]
        ),
        "pre_final_actual_champion_probability": float(
            champion_before_final["champion"]
        ),
        "highest_actual_champion_probability_before_final": float(
            snapshot_summary.iloc[:-1][
                "actual_champion_probability"
            ].max()
        )
        if len(snapshot_summary) > 1
        else float(initial_summary["actual_champion_probability"]),
        "final_pre_match_odds": [
            {
                "team": str(row["team"]),
                "champion_probability": float(row["champion"]),
            }
            for _, row in pre_final.head(2).iterrows()
        ],
    }

    return history, snapshot_summary, final_report


# ============================================================
# Prediction Archive Loading and Legacy Migration
# ============================================================


def normalize_history_dataframe(history: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "snapshot_label",
        "team",
        "quarterfinal",
        "semifinal",
        "final",
        "champion",
    }

    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        raise ValueError(
            "prediction_history.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    normalized = history.copy()

    if "checkpoint_index" not in normalized.columns:
        labels = normalized["snapshot_label"].astype(str).drop_duplicates()
        label_to_index = {
            label: index for index, label in enumerate(labels.tolist())
        }
        normalized["checkpoint_index"] = normalized[
            "snapshot_label"
        ].map(label_to_index)

    normalized["checkpoint_index"] = pd.to_numeric(
        normalized["checkpoint_index"],
        errors="raise",
    ).astype(int)

    defaults: dict[str, Any] = {
        "snapshot_id": "",
        "snapshot_date": "",
        "snapshot_short_label": "",
        "trigger_match_id": "",
        "trigger_result": "",
    }

    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default

    if "champion_rank" not in normalized.columns:
        normalized["champion_rank"] = normalized.groupby(
            "checkpoint_index"
        )["champion"].rank(method="min", ascending=False)

    normalized["champion_rank"] = pd.to_numeric(
        normalized["champion_rank"],
        errors="coerce",
    ).fillna(len(INITIAL_TEAMS)).astype(int)

    for probability_column in [
        "quarterfinal",
        "semifinal",
        "final",
        "champion",
    ]:
        normalized[probability_column] = pd.to_numeric(
            normalized[probability_column],
            errors="raise",
        )

    return normalized.sort_values(
        ["checkpoint_index", "champion", "team"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def derive_snapshot_summary(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for checkpoint_index, group in history.groupby(
        "checkpoint_index",
        sort=True,
    ):
        ranked = group.sort_values("champion", ascending=False)
        top_row = ranked.iloc[0]
        champion_matches = ranked[ranked["team"] == ACTUAL_CHAMPION]

        if champion_matches.empty:
            champion_probability = 0.0
            champion_rank = len(INITIAL_TEAMS)
        else:
            champion_row = champion_matches.iloc[0]
            champion_probability = float(champion_row["champion"])
            champion_rank = int(champion_row["champion_rank"])

        label = str(top_row.get("snapshot_label", f"Checkpoint {checkpoint_index}"))
        short_label = str(top_row.get("snapshot_short_label", ""))
        if not short_label:
            short_label = (
                "Start"
                if int(checkpoint_index) == 0
                else f"Checkpoint {int(checkpoint_index)}"
            )

        rows.append(
            {
                "checkpoint_index": int(checkpoint_index),
                "snapshot_id": str(top_row.get("snapshot_id", "")),
                "snapshot_date": str(top_row.get("snapshot_date", "")),
                "snapshot_label": label,
                "snapshot_short_label": short_label,
                "trigger_match_id": str(top_row.get("trigger_match_id", "")),
                "trigger_result": str(top_row.get("trigger_result", "")),
                "teams_remaining": int((ranked["champion"] > 0).sum()),
                "top_projected_champion": str(top_row["team"]),
                "top_champion_probability": float(top_row["champion"]),
                "actual_champion": ACTUAL_CHAMPION,
                "actual_champion_probability": champion_probability,
                "actual_champion_rank": champion_rank,
            }
        )

    return pd.DataFrame(rows).sort_values("checkpoint_index")


def load_prediction_archive(
    history_path: Path = PREDICTION_HISTORY_PATH,
    summary_path: Path = SNAPSHOT_SUMMARY_PATH,
    report_path: Path = FINAL_MODEL_REPORT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not history_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}

    history = normalize_history_dataframe(pd.read_csv(history_path))

    if summary_path.exists():
        snapshot_summary = pd.read_csv(summary_path)
        if "checkpoint_index" not in snapshot_summary.columns:
            snapshot_summary = derive_snapshot_summary(history)
        else:
            snapshot_summary["checkpoint_index"] = pd.to_numeric(
                snapshot_summary["checkpoint_index"],
                errors="raise",
            ).astype(int)
            snapshot_summary = snapshot_summary.sort_values(
                "checkpoint_index"
            ).reset_index(drop=True)
    else:
        snapshot_summary = derive_snapshot_summary(history)

    final_report = load_json(report_path) if report_path.exists() else {}
    return history, snapshot_summary, final_report


# ============================================================
# Streamlit Cache Wrappers
# ============================================================


@st.cache_resource
def load_interactive_resources() -> tuple[Any, list[str], pd.DataFrame]:
    model, features = load_model_and_features()
    matches = load_matches()
    return model, features, matches


@st.cache_data(show_spinner=False)
def run_cached_scenario(
    state_json: str,
    num_simulations: int,
    seed: int,
) -> pd.DataFrame:
    model, features, matches = load_interactive_resources()
    state = json.loads(state_json)

    return run_live_simulations(
        state=state,
        model=model,
        features=features,
        matches=matches,
        num_simulations=num_simulations,
        seed=seed,
    )


# ============================================================
# UI Helpers
# ============================================================


def probability_to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def display_top_three(results: pd.DataFrame) -> None:
    columns = st.columns(3)

    for column, (team, row) in zip(columns, results.head(3).iterrows()):
        column.metric(
            label=str(team),
            value=probability_to_percent(float(row["champion"])),
            delta="Champion odds",
        )


def display_completed_matches(state: dict[str, Any]) -> None:
    completed = state.get("completed_matches", [])

    if not completed:
        st.info("No official results were locked at this checkpoint.")
        return

    completed_frame = pd.DataFrame(completed)
    completed_frame["round"] = completed_frame["round"].map(
        format_round_name
    )

    st.dataframe(
        completed_frame[
            [
                "date",
                "round",
                "home_team",
                "home_score",
                "away_score",
                "away_team",
                "winner",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def make_probability_bar_chart(
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

    figure = px.bar(
        chart_data,
        x="team",
        y=column_name,
        title=title,
        labels={
            "team": "Team",
            column_name: "Probability (%)",
        },
    )
    figure.update_layout(
        xaxis_tickangle=-30,
        height=450,
    )
    return figure


def make_timeline_chart(
    history: pd.DataFrame,
    snapshot_summary: pd.DataFrame,
    teams: list[str],
):
    chart_data = history[history["team"].isin(teams)].copy()
    chart_data["champion_percent"] = chart_data["champion"] * 100

    figure = px.line(
        chart_data,
        x="checkpoint_index",
        y="champion_percent",
        color="team",
        markers=True,
        custom_data=["snapshot_label", "trigger_result"],
        labels={
            "checkpoint_index": "Prediction checkpoint",
            "champion_percent": "Champion probability (%)",
            "team": "Team",
        },
        title="Champion Probability Through the Knockout Stage",
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "%{customdata[0]}<br>"
            "Champion probability: %{y:.2f}%<extra></extra>"
        )
    )

    tick_values = snapshot_summary["checkpoint_index"].tolist()
    if "snapshot_short_label" in snapshot_summary.columns:
        tick_text = snapshot_summary["snapshot_short_label"].astype(str).tolist()
    else:
        tick_text = [f"#{value}" for value in tick_values]

    figure.update_xaxes(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        tickangle=-35,
    )
    figure.update_layout(height=540)
    return figure


def initialize_scenario_state(
    official_state: dict[str, Any],
    checkpoint_index: int,
) -> None:
    scenario_key = f"checkpoint_{checkpoint_index}"

    if st.session_state.get("scenario_start_key") != scenario_key:
        st.session_state["scenario_start_key"] = scenario_key
        st.session_state["scenario_state"] = get_state_at_checkpoint(
            official_state,
            checkpoint_index,
        )
        st.session_state["scenario_path"] = []


def reset_scenario_state(
    official_state: dict[str, Any],
    checkpoint_index: int,
) -> None:
    st.session_state["scenario_state"] = get_state_at_checkpoint(
        official_state,
        checkpoint_index,
    )
    st.session_state["scenario_path"] = []


def render_model_load_error(error: Exception) -> None:
    st.error("The interactive model could not be loaded.")
    st.code(str(error))
    st.info(
        "The permanent archive still works. The What-If Lab requires the "
        "package versions in requirements.txt, especially scikit-learn 1.9.0."
    )


# ============================================================
# Main Streamlit App
# ============================================================


def main() -> None:
    st.set_page_config(
        page_title="FIFA World Cup 2026 Prediction Archive",
        page_icon="⚽",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .champion-card {
            border: 2px solid currentColor;
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin: 0.8rem 0 1.25rem 0;
            text-align: center;
        }
        .champion-title {
            font-size: 2.15rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .champion-subtitle {
            font-size: 1.05rem;
            opacity: 0.82;
        }
        .archive-note {
            border-left: 4px solid currentColor;
            padding: 0.15rem 0 0.15rem 1rem;
            margin: 0.75rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        official_state = load_official_state()
    except Exception as error:
        st.error("The official results file contains an error.")
        st.code(str(error))
        st.stop()

    try:
        tournament_summary = load_json(TOURNAMENT_SUMMARY_PATH)
    except Exception as error:
        st.error("The tournament summary file could not be loaded.")
        st.code(str(error))
        st.stop()

    try:
        history, snapshot_summary, final_report = load_prediction_archive()
    except Exception as error:
        st.error("The saved prediction history contains an error.")
        st.code(str(error))
        st.info(
            "Run `python scripts/build_prediction_history.py --simulations 10000` "
            "to rebuild the archive from the official result sequence."
        )
        history = pd.DataFrame()
        snapshot_summary = pd.DataFrame()
        final_report = {}

    champion = str(tournament_summary["champion"])
    runner_up = str(tournament_summary["runner_up"])
    third_place = str(tournament_summary["third_place"])
    fourth_place = str(tournament_summary["fourth_place"])
    final_data = tournament_summary["final"]

    st.title("⚽ FIFA World Cup 2026 Prediction Archive")
    st.caption(
        "The completed record of a live machine learning experiment: the "
        "champion, official results, and every reconstructed prediction "
        "checkpoint from the knockout stage."
    )

    st.markdown(
        f"""
        <div class="champion-card">
            <div class="champion-title">🏆 {champion} — 2026 World Champions</div>
            <div class="champion-subtitle">
                {final_data['home_team']} {final_data['home_score']}-{final_data['away_score']} {final_data['away_team']} {final_data.get('decision', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_columns = st.columns(4)
    hero_columns[0].metric("Champion", champion)
    hero_columns[1].metric("Runner-up", runner_up)
    hero_columns[2].metric("Third place", third_place)
    hero_columns[3].metric(
        "Saved checkpoints",
        0 if snapshot_summary.empty else len(snapshot_summary),
    )

    if history.empty or snapshot_summary.empty:
        st.warning(
            "The final result is loaded, but the permanent probability history "
            "has not been rebuilt in the new format yet."
        )
        st.code(
            "source .venv/bin/activate\n"
            "python scripts/build_prediction_history.py --simulations 10000"
        )

    overview_tab, timeline_tab, snapshot_tab, results_tab, scenario_tab, about_tab = st.tabs(
        [
            "Final Overview",
            "Prediction Timeline",
            "Snapshot Explorer",
            "Official Results",
            "What-If Lab",
            "About the Model",
        ]
    )

    # --------------------------------------------------------
    # Final Overview
    # --------------------------------------------------------

    with overview_tab:
        st.subheader("The final outcome and what the model said beforehand")

        placement_columns = st.columns(4)
        placement_columns[0].metric("1st", champion)
        placement_columns[1].metric("2nd", runner_up)
        placement_columns[2].metric("3rd", third_place)
        placement_columns[3].metric("4th", fourth_place)

        st.success(
            f"{final_data['home_team']} {final_data['home_score']}-"
            f"{final_data['away_score']} {final_data['away_team']} — "
            f"{champion} won {final_data.get('decision', '').strip()}."
        )

        if not history.empty and not snapshot_summary.empty:
            pre_final_index = int(snapshot_summary["checkpoint_index"].max()) - 1
            pre_final = history[
                history["checkpoint_index"] == pre_final_index
            ].sort_values("champion", ascending=False)

            st.markdown("### Model odds immediately before the final")
            pre_final_columns = st.columns(2)
            for column, (_, row) in zip(
                pre_final_columns,
                pre_final.head(2).iterrows(),
            ):
                column.metric(
                    str(row["team"]),
                    probability_to_percent(float(row["champion"])),
                    "Chance to win the final",
                )

            champion_history = history[
                history["team"] == ACTUAL_CHAMPION
            ].sort_values("checkpoint_index")
            initial_champion = champion_history.iloc[0]
            before_final_champion = champion_history[
                champion_history["checkpoint_index"] == pre_final_index
            ].iloc[0]
            peak_before_final = champion_history[
                champion_history["checkpoint_index"] < int(
                    snapshot_summary["checkpoint_index"].max()
                )
            ]["champion"].max()

            st.markdown("### Spain's prediction journey at a glance")
            journey_columns = st.columns(3)
            journey_columns[0].metric(
                "Initial Spain odds",
                probability_to_percent(float(initial_champion["champion"])),
                f"Rank #{int(initial_champion['champion_rank'])}",
            )
            journey_columns[1].metric(
                "Peak before the final",
                probability_to_percent(float(peak_before_final)),
            )
            journey_columns[2].metric(
                "Spain odds before final",
                probability_to_percent(
                    float(before_final_champion["champion"])
                ),
            )

            default_overview_teams = [
                team
                for team in [
                    "Spain",
                    "Argentina",
                    "France",
                    "England",
                    "Brazil",
                ]
                if team in history["team"].unique()
            ]
            overview_figure = make_timeline_chart(
                history,
                snapshot_summary,
                default_overview_teams,
            )
            st.plotly_chart(overview_figure, width="stretch")

            initial_summary = snapshot_summary.iloc[0]
            st.markdown("### Forecast accountability")
            st.markdown(
                f"""
                <div class="archive-note">
                At the first saved checkpoint, the model's top projected champion was
                <strong>{initial_summary['top_projected_champion']}</strong> at
                <strong>{probability_to_percent(float(initial_summary['top_champion_probability']))}</strong>.
                The eventual champion, <strong>{champion}</strong>, started at
                <strong>{probability_to_percent(float(initial_summary['actual_champion_probability']))}</strong>
                and rank <strong>#{int(initial_summary['actual_champion_rank'])}</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Prediction Timeline
    # --------------------------------------------------------

    with timeline_tab:
        st.subheader("Champion probability at every saved checkpoint")
        st.write(
            "The graph no longer depends on Streamlit session state. It reads "
            "a committed CSV generated by replaying each official result one "
            "match at a time. That is why the historical line now survives "
            "restarts and deployments."
        )

        if history.empty or snapshot_summary.empty:
            st.info(
                "Build the archive first with `python scripts/build_prediction_history.py --simulations 10000`."
            )
        else:
            all_teams = sorted(history["team"].unique().tolist())
            default_teams = [
                team
                for team in [
                    "Spain",
                    "Argentina",
                    "France",
                    "England",
                    "Brazil",
                ]
                if team in all_teams
            ]

            selected_teams = st.multiselect(
                "Teams shown on the timeline",
                options=all_teams,
                default=default_teams,
            )

            if selected_teams:
                timeline_figure = make_timeline_chart(
                    history,
                    snapshot_summary,
                    selected_teams,
                )
                st.plotly_chart(timeline_figure, width="stretch")
            else:
                st.info("Select at least one team.")

            st.markdown("### Spain's complete checkpoint history")
            champion_history = history[
                history["team"] == ACTUAL_CHAMPION
            ][
                [
                    "checkpoint_index",
                    "snapshot_date",
                    "snapshot_label",
                    "champion",
                    "champion_rank",
                ]
            ].copy()
            champion_history["champion_probability_percent"] = (
                champion_history["champion"] * 100
            ).round(2)

            st.dataframe(
                champion_history[
                    [
                        "checkpoint_index",
                        "snapshot_date",
                        "snapshot_label",
                        "champion_probability_percent",
                        "champion_rank",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Download the archive")
            download_columns = st.columns(3)
            download_columns[0].download_button(
                "Download all prediction rows",
                data=history.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
            download_columns[1].download_button(
                "Download checkpoint summary",
                data=snapshot_summary.to_csv(index=False),
                file_name="prediction_snapshot_summary.csv",
                mime="text/csv",
            )
            download_columns[2].download_button(
                "Download official results",
                data=json.dumps(official_state, indent=2),
                file_name="official_results.json",
                mime="application/json",
            )

    # --------------------------------------------------------
    # Snapshot Explorer
    # --------------------------------------------------------

    with snapshot_tab:
        st.subheader("Open any historical prediction snapshot")

        if history.empty or snapshot_summary.empty:
            st.info("No rebuilt snapshots are available yet.")
        else:
            checkpoint_options = {
                str(row["snapshot_label"]): int(row["checkpoint_index"])
                for _, row in snapshot_summary.iterrows()
            }
            selected_label = st.selectbox(
                "Prediction checkpoint",
                options=list(checkpoint_options.keys()),
                index=max(len(checkpoint_options) - 2, 0),
            )
            selected_index = checkpoint_options[selected_label]

            snapshot = history[
                history["checkpoint_index"] == selected_index
            ].sort_values("champion", ascending=False)

            snapshot_columns = st.columns(3)
            top_three = snapshot.head(3)
            for column, (_, row) in zip(
                snapshot_columns,
                top_three.iterrows(),
            ):
                column.metric(
                    str(row["team"]),
                    probability_to_percent(float(row["champion"])),
                    "Champion odds",
                )

            snapshot_chart = snapshot.head(10).copy()
            snapshot_chart["champion_percent"] = (
                snapshot_chart["champion"] * 100
            )
            snapshot_figure = px.bar(
                snapshot_chart,
                x="team",
                y="champion_percent",
                title=selected_label,
                labels={
                    "team": "Team",
                    "champion_percent": "Champion probability (%)",
                },
            )
            snapshot_figure.update_layout(
                xaxis_tickangle=-30,
                height=450,
            )
            st.plotly_chart(snapshot_figure, width="stretch")

            snapshot_table = snapshot[
                [
                    "team",
                    "quarterfinal",
                    "semifinal",
                    "final",
                    "champion",
                    "champion_rank",
                ]
            ].copy()
            for probability_column in [
                "quarterfinal",
                "semifinal",
                "final",
                "champion",
            ]:
                snapshot_table[probability_column] = (
                    snapshot_table[probability_column] * 100
                ).round(2)

            st.dataframe(
                snapshot_table,
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Results known at that moment")
            checkpoint_state = get_state_at_checkpoint(
                official_state,
                selected_index,
            )
            display_completed_matches(checkpoint_state)

            available_matches = get_available_matches(checkpoint_state)
            if available_matches:
                st.markdown("### Next model matchup from that snapshot")
                earliest_date = min(
                    pd.Timestamp(match["date"])
                    for match, _, _ in available_matches
                )
                next_matches = [
                    item
                    for item in available_matches
                    if pd.Timestamp(item[0]["date"]) == earliest_date
                ]

                try:
                    model, features, matches = load_interactive_resources()
                    profiles = build_live_profiles(matches, checkpoint_state)
                    recent_forms = build_recent_forms(matches, checkpoint_state)
                    prediction_cache: dict[
                        tuple[Any, ...], dict[str, float]
                    ] = {}

                    for match, home_team, away_team in next_matches:
                        probabilities = predict_live_match(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=pd.Timestamp(match["date"]),
                            model=model,
                            features=features,
                            matches=matches,
                            profiles=profiles,
                            recent_forms=recent_forms,
                            prediction_cache=prediction_cache,
                            state_key=state_signature(checkpoint_state),
                            neutral=True,
                        )

                        home_win = probabilities.get("home_win", 0.0)
                        draw = probabilities.get("draw", 0.0)
                        away_win = probabilities.get("away_win", 0.0)
                        knockout_total = home_win + away_win
                        home_advance = (
                            0.5
                            if knockout_total <= 0
                            else home_win / knockout_total
                        )
                        away_advance = 1 - home_advance

                        st.write(
                            f"**{format_round_name(match['round'])}: "
                            f"{home_team} vs {away_team}**"
                        )
                        probability_frame = pd.DataFrame(
                            [
                                {
                                    "Outcome": f"{home_team} win",
                                    "Probability (%)": round(home_win * 100, 2),
                                },
                                {
                                    "Outcome": "Draw",
                                    "Probability (%)": round(draw * 100, 2),
                                },
                                {
                                    "Outcome": f"{away_team} win",
                                    "Probability (%)": round(away_win * 100, 2),
                                },
                                {
                                    "Outcome": f"{home_team} advances",
                                    "Probability (%)": round(home_advance * 100, 2),
                                },
                                {
                                    "Outcome": f"{away_team} advances",
                                    "Probability (%)": round(away_advance * 100, 2),
                                },
                            ]
                        )
                        st.dataframe(
                            probability_frame,
                            width="stretch",
                            hide_index=True,
                        )
                except Exception as error:
                    render_model_load_error(error)

    # --------------------------------------------------------
    # Official Results
    # --------------------------------------------------------

    with results_tab:
        st.subheader("Final tournament record")

        result_columns = st.columns(4)
        result_columns[0].metric("1st", champion)
        result_columns[1].metric("2nd", runner_up)
        result_columns[2].metric("3rd", third_place)
        result_columns[3].metric("4th", fourth_place)

        st.markdown("### Championship match")
        st.success(
            f"{final_data['home_team']} {final_data['home_score']}-"
            f"{final_data['away_score']} {final_data['away_team']} — "
            f"{champion} won {final_data.get('decision', '').strip()}."
        )

        bronze_match = tournament_summary.get("third_place_match")
        if isinstance(bronze_match, dict):
            st.markdown("### Third-place match")
            st.info(
                f"{bronze_match['home_team']} {bronze_match['home_score']}-"
                f"{bronze_match['away_score']} {bronze_match['away_team']} — "
                f"{bronze_match['winner']} finished third."
            )

        st.markdown("### Official result path used by the model archive")
        display_completed_matches(official_state)

        st.caption(
            "The third-place playoff is displayed above but is not part of the "
            "champion-probability bracket, so it is stored in "
            "tournament_summary.json rather than official_results.json."
        )

    # --------------------------------------------------------
    # What-If Lab
    # --------------------------------------------------------

    with scenario_tab:
        st.subheader("Historical What-If Lab")
        st.write(
            "Start from any saved checkpoint, change the remaining results, "
            "and rerun the model. These hypothetical changes only live in your "
            "browser session and reset on refresh."
        )

        if snapshot_summary.empty:
            st.info("Build the prediction history before using this section.")
        else:
            start_labels = snapshot_summary["snapshot_label"].astype(str).tolist()
            selected_start_label = st.selectbox(
                "Start from checkpoint",
                start_labels,
                index=0,
                key="scenario_start_selector",
            )
            selected_start_index = int(
                snapshot_summary.loc[
                    snapshot_summary["snapshot_label"].astype(str)
                    == selected_start_label,
                    "checkpoint_index",
                ].iloc[0]
            )

            initialize_scenario_state(
                official_state,
                selected_start_index,
            )

            settings_columns = st.columns(3)
            num_simulations = int(
                settings_columns[0].selectbox(
                    "Scenario simulations",
                    [500, 1000, 2500, 5000, 10000],
                    index=1,
                )
            )
            seed = int(
                settings_columns[1].number_input(
                    "Random seed",
                    min_value=1,
                    max_value=999999,
                    value=42,
                    step=1,
                )
            )
            if settings_columns[2].button("Reset this scenario"):
                reset_scenario_state(
                    official_state,
                    selected_start_index,
                )
                st.rerun()

            scenario_state = st.session_state["scenario_state"]

            try:
                with st.spinner("Running the temporary scenario..."):
                    scenario_results = run_cached_scenario(
                        state_signature(scenario_state),
                        num_simulations,
                        seed,
                    )
            except Exception as error:
                render_model_load_error(error)
                scenario_results = pd.DataFrame()

            if not scenario_results.empty:
                display_top_three(scenario_results)

                scenario_figure = make_probability_bar_chart(
                    scenario_results,
                    "champion",
                    "Temporary Scenario Champion Odds",
                )
                st.plotly_chart(scenario_figure, width="stretch")

                st.markdown("### Results locked into this scenario")
                display_completed_matches(scenario_state)

                available_matches = get_available_matches(scenario_state)

                if not available_matches:
                    st.success("This scenario has reached a champion.")
                else:
                    earliest_date = min(
                        pd.Timestamp(match["date"])
                        for match, _, _ in available_matches
                    )
                    next_matches = [
                        item
                        for item in available_matches
                        if pd.Timestamp(item[0]["date"]) == earliest_date
                    ]

                    option_map = {
                        format_match_label(match, home_team, away_team): (
                            match,
                            home_team,
                            away_team,
                        )
                        for match, home_team, away_team in next_matches
                    }

                    selected_match_label = st.selectbox(
                        "Choose the next hypothetical result",
                        list(option_map.keys()),
                    )
                    selected_match, home_team, away_team = option_map[
                        selected_match_label
                    ]

                    with st.form("scenario_result_form"):
                        score_columns = st.columns(2)
                        home_score = score_columns[0].number_input(
                            f"{home_team} score",
                            min_value=0,
                            max_value=20,
                            value=0,
                            step=1,
                        )
                        away_score = score_columns[1].number_input(
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

                        submitted = st.form_submit_button(
                            "Apply temporary result"
                        )

                    if submitted:
                        try:
                            st.session_state["scenario_state"] = (
                                add_completed_match_to_state(
                                    state=scenario_state,
                                    match=selected_match,
                                    home_team=home_team,
                                    away_team=away_team,
                                    match_date=selected_match["date"],
                                    home_score=int(home_score),
                                    away_score=int(away_score),
                                    winner=winner,
                                )
                            )
                            st.session_state["scenario_path"].append(
                                f"{home_team} {int(home_score)}-"
                                f"{int(away_score)} {away_team}"
                            )
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

                if st.session_state.get("scenario_path"):
                    st.markdown("### Temporary scenario path")
                    for index, result in enumerate(
                        st.session_state["scenario_path"],
                        start=1,
                    ):
                        st.write(f"{index}. {result}")

    # --------------------------------------------------------
    # About the Model
    # --------------------------------------------------------

    with about_tab:
        st.subheader("About this project")
        st.write(
            "This began as a learning-first machine learning project and ended "
            "as a deployed forecasting archive. The model predicts home win, "
            "draw, and away win probabilities. The tournament layer removes "
            "the draw probability for knockout advancement and runs Monte Carlo "
            "simulations to estimate each team's chance of reaching later rounds."
        )

        st.markdown("### Final model selection")
        st.dataframe(
            MODEL_RESULTS,
            width="stretch",
            hide_index=True,
        )

        st.markdown("### Features used")
        try:
            _, saved_features, _ = load_interactive_resources()
            st.code("\n".join(saved_features))
        except Exception:
            st.code(
                "elo_difference\n"
                "home_advantage\n"
                "recent_win_rate_difference\n"
                "recent_goals_for_difference\n"
                "recent_goals_against_difference\n"
                "recent_goal_difference_difference\n"
                "rest_days_difference\n"
                "streak_difference\n"
                "is_world_cup\n"
                "is_qualification\n"
                "is_friendly\n"
                "tournament_importance\n"
                "head_to_head_difference\n"
                "attack_rating_difference\n"
                "defense_rating_difference"
            )

        st.markdown("### Why the graph now works")
        st.write(
            "The old graph expected historical rows to already exist, but the "
            "public app never permanently wrote each new snapshot. Streamlit "
            "session state disappears after restarts, so the line could not "
            "build a durable timeline. The new history script replays the final "
            "official result list from zero completed matches through the final, "
            "runs the model at every prefix, and commits the resulting CSV."
        )

        st.markdown("### Leakage protection for the archive")
        st.write(
            "Historical checkpoints use only processed match rows dated before "
            "July 5, 2026. Each later tournament result is then applied from "
            "official_results.json in chronological order. This prevents a "
            "final, fully updated CSV from leaking later results into earlier "
            "prediction snapshots."
        )

        if final_report:
            st.markdown("### Reproducibility report")
            st.json(final_report)

        st.markdown("### Limitations")
        st.markdown(
            """
            - The model does not include injuries, starting lineups, betting markets, or player-level data.
            - Official results update team profiles and bracket state; they do not retrain the classifier.
            - Knockout draws are handled by removing draw probability and normalizing the two win probabilities.
            - This is a portfolio forecasting experiment, not betting advice or a claim of certainty.
            """
        )


if __name__ == "__main__":
    main()
