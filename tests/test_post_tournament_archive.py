from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


def _load_app_module():
    """Import app.py without requiring a running Streamlit process."""
    streamlit_stub = types.ModuleType("streamlit")

    def passthrough_decorator(*decorator_args, **decorator_kwargs):
        if (
            decorator_args
            and callable(decorator_args[0])
            and len(decorator_args) == 1
            and not decorator_kwargs
        ):
            return decorator_args[0]

        def decorator(function):
            return function

        return decorator

    streamlit_stub.cache_resource = passthrough_decorator
    streamlit_stub.cache_data = passthrough_decorator
    sys.modules.setdefault("streamlit", streamlit_stub)

    specification = importlib.util.spec_from_file_location(
        "fifa_archive_app",
        APP_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification is not None
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


APP = _load_app_module()


class DummyModel:
    classes_ = np.array(["away_win", "draw", "home_win"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        elo_difference = float(X.iloc[0]["elo_difference"])
        raw_home_probability = 1 / (
            1 + np.exp(-(elo_difference / 400.0))
        )
        draw_probability = 0.20
        home_probability = raw_home_probability * 0.80
        away_probability = 0.80 - home_probability
        return np.array(
            [[away_probability, draw_probability, home_probability]]
        )


FEATURES = [
    "elo_difference",
    "home_advantage",
    "recent_win_rate_difference",
    "recent_goals_for_difference",
    "recent_goals_against_difference",
    "recent_goal_difference_difference",
    "rest_days_difference",
    "streak_difference",
    "is_world_cup",
    "is_qualification",
    "is_friendly",
    "tournament_importance",
    "head_to_head_difference",
    "attack_rating_difference",
    "defense_rating_difference",
]


def make_matches() -> pd.DataFrame:
    rows = []
    teams = APP.INITIAL_TEAMS

    for index, team in enumerate(teams):
        opponent = teams[(index + 1) % len(teams)]
        rows.append(
            {
                "date": pd.Timestamp("2026-07-01")
                - pd.Timedelta(days=index % 5),
                "home_team": team,
                "away_team": opponent,
                "home_score": 1 + (index % 2),
                "away_score": index % 2,
                "home_elo_before": 1450 + index * 15,
                "away_elo_before": (
                    1450 + ((index + 1) % len(teams)) * 15
                ),
                "home_attack_before": 1.0 + index * 0.01,
                "away_attack_before": (
                    1.0 + ((index + 1) % len(teams)) * 0.01
                ),
                "home_defense_before": 1.0,
                "away_defense_before": 1.0,
                "home_streak": 0,
                "away_streak": 0,
            }
        )

    # This future row must never affect an earlier checkpoint.
    rows.append(
        {
            "date": pd.Timestamp("2026-07-20"),
            "home_team": "Spain",
            "away_team": "Argentina",
            "home_score": 9,
            "away_score": 0,
            "home_elo_before": 9999,
            "away_elo_before": 1,
            "home_attack_before": 99,
            "away_attack_before": 0,
            "home_defense_before": 99,
            "away_defense_before": 0,
            "home_streak": 99,
            "away_streak": -99,
        }
    )

    return pd.DataFrame(rows)


def load_official_state() -> dict:
    return APP.load_json(ROOT / "data" / "app" / "official_results.json")


def test_final_official_state_is_valid() -> None:
    state = load_official_state()
    APP.validate_state(state)

    assert len(state["completed_matches"]) == 13
    assert state["completed_matches"][-1]["match_id"] == "final"
    assert state["completed_matches"][-1]["winner"] == "Spain"


def test_future_processed_rows_are_excluded_from_base_snapshot() -> None:
    matches = make_matches()
    base_matches = APP.get_base_matches(matches)
    profile = APP.get_base_team_profile(base_matches, "Spain")

    assert base_matches["date"].max() < APP.TRACKING_START_DATE
    assert profile["elo"] != 9999
    assert profile["attack"] != 99


def test_history_contains_every_checkpoint_and_team() -> None:
    history, summary, report = APP.build_prediction_history(
        official_state=load_official_state(),
        model=DummyModel(),
        features=FEATURES,
        matches=make_matches(),
        num_simulations=50,
        seed=42,
    )

    assert len(summary) == 14
    assert len(history) == 14 * len(APP.INITIAL_TEAMS)
    assert report["total_checkpoints"] == 14
    assert history["checkpoint_index"].nunique() == 14


def test_final_snapshot_locks_spain_at_one_hundred_percent() -> None:
    history, _, _ = APP.build_prediction_history(
        official_state=load_official_state(),
        model=DummyModel(),
        features=FEATURES,
        matches=make_matches(),
        num_simulations=25,
        seed=42,
    )

    final_snapshot = history[
        history["checkpoint_index"]
        == history["checkpoint_index"].max()
    ]
    spain_probability = float(
        final_snapshot.loc[
            final_snapshot["team"] == "Spain",
            "champion",
        ].iloc[0]
    )

    assert spain_probability == 1.0
    assert final_snapshot["champion"].sum() == 1.0
