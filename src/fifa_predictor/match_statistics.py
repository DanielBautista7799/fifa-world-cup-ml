def determine_result(home_score: int, away_score: int) -> str:
    if home_score<0 or away_score <0:
        raise ValueError("Scores cannot be negative")

    if home_score>away_score:
        return "home_win"

    if away_score>home_score:
        return "away_win"

    return "draw"

def calculate_points(home_score: int, away_score: int) -> tuple[int, int]:
        result = determine_result(home_score,away_score)
        if result == "home_win":
            return (3,0)
        if result == "away_win":
            return (0,3)
        if result == "draw":
            return (1,1)

def calculate_team_record(matches: list[dict], team: str) -> dict:
    record = {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
    }

    for match in matches:
        if match["home_team"] == team:
            goals_for = match["home_score"]
            goals_against = match["away_score"]

        elif match["away_team"] == team:
            goals_for = match["away_score"]
            goals_against = match["home_score"]

        else:
            continue

        record["played"] += 1
        record["goals_for"] += goals_for
        record["goals_against"] += goals_against

        if goals_for > goals_against:
            record["wins"] += 1
            record["points"] += 3

        elif goals_for == goals_against:
            record["draws"] += 1
            record["points"] += 1

        else:
            record["losses"] += 1

    record["goal_difference"] = (
        record["goals_for"] - record["goals_against"]
    )

    return record