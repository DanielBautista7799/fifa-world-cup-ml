import pytest

from fifa_predictor.match_statistics import (
    calculate_points,
    calculate_team_record,
    determine_result,
)
MATCHES = [
    {
        "home_team": "Argentina",
        "away_team": "Brazil",
        "home_score": 2,
        "away_score": 1,
    },
    {
        "home_team": "France",
        "away_team": "Argentina",
        "home_score": 1,
        "away_score": 1,
    },
    {
        "home_team": "Argentina",
        "away_team": "Japan",
        "home_score": 0,
        "away_score": 2,
    },
    {
        "home_team": "Brazil",
        "away_team": "Argentina",
        "home_score": 0,
        "away_score": 3,
    },
]

def test_home_team_wins():
    result = determine_result(3,1)
    assert result == "home_win"

def test_away_team_wins():
    result = determine_result(0,2)
    assert result == "away_win"
    
def test_match_is_draw():
    result = determine_result(2, 2)

    assert result == "draw"


def test_zero_zero_is_draw():
    result = determine_result(0, 0)

    assert result == "draw"


def test_negative_score_raises_error():
        with pytest.raises(ValueError):
            determine_result(-1,2)


def test_home_team_gets_three_points():
    assert calculate_points(2, 0) == (3, 0)


def test_away_team_gets_three_points():
    assert calculate_points(1, 4) == (0, 3)


def test_both_teams_get_one_point_for_draw():
    assert calculate_points(2, 2) == (1, 1)


def test_zero_zero_gives_both_teams_one_point():
    assert calculate_points(0, 0) == (1, 1)


def test_calculate_points_rejects_negative_scores():
    with pytest.raises(ValueError):
        calculate_points(2, -1)

def test_calculate_argentina_record():
    result = calculate_team_record(MATCHES, "Argentina")

    assert result == {
        "played": 4,
        "wins": 2,
        "draws": 1,
        "losses": 1,
        "goals_for": 6,
        "goals_against": 4,
        "goal_difference": 2,
        "points": 7,
    }