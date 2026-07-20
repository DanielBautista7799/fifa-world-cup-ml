from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


HISTORY_PATH = ROOT / "data" / "app" / "prediction_history.csv"
SUMMARY_PATH = ROOT / "data" / "app" / "prediction_snapshot_summary.csv"
REPORT_PATH = ROOT / "data" / "app" / "final_model_report.json"
LEGACY_BACKUP_PATH = (
    ROOT / "data" / "app" / "prediction_history_before_rebuild.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild every World Cup prediction checkpoint from the final "
            "official result sequence."
        )
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10_000,
        help="Monte Carlo simulations per checkpoint (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used at every checkpoint (default: 42).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not preserve the existing prediction_history.csv first.",
    )
    return parser.parse_args()


def backup_existing_history() -> None:
    if not HISTORY_PATH.exists() or LEGACY_BACKUP_PATH.exists():
        return

    shutil.copy2(HISTORY_PATH, LEGACY_BACKUP_PATH)
    print(
        "Preserved the existing history file at "
        f"{LEGACY_BACKUP_PATH.relative_to(ROOT)}"
    )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def write_json_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()

    if args.simulations <= 0:
        raise SystemExit("--simulations must be greater than zero.")

    if args.seed <= 0:
        raise SystemExit("--seed must be greater than zero.")

    print("Loading official result sequence...")
    official_state = app.load_official_state()
    app.validate_state(official_state)

    expected_matches = len(app.BRACKET_MATCHES)
    completed_matches = len(official_state["completed_matches"])

    if completed_matches != expected_matches:
        raise SystemExit(
            "The tournament archive is incomplete. "
            f"Expected {expected_matches} completed bracket matches, "
            f"found {completed_matches}."
        )

    final_record = official_state["completed_matches"][-1]
    if final_record["match_id"] != "final":
        raise SystemExit("The final result must be the last official record.")

    if final_record["winner"] != app.ACTUAL_CHAMPION:
        raise SystemExit(
            f"Expected final champion {app.ACTUAL_CHAMPION}, "
            f"found {final_record['winner']}."
        )

    if not args.no_backup:
        backup_existing_history()

    print("Loading the saved model, feature list, and processed match data...")
    model, features = app.load_model_and_features()
    matches = app.load_matches()

    print(
        f"Building {completed_matches + 1} checkpoints with "
        f"{args.simulations:,} simulations per checkpoint..."
    )
    history, snapshot_summary, final_report = app.build_prediction_history(
        official_state=official_state,
        model=model,
        features=features,
        matches=matches,
        num_simulations=args.simulations,
        seed=args.seed,
    )

    expected_history_rows = (
        completed_matches + 1
    ) * len(app.INITIAL_TEAMS)

    if len(history) != expected_history_rows:
        raise RuntimeError(
            f"Expected {expected_history_rows} history rows, "
            f"generated {len(history)}."
        )

    if len(snapshot_summary) != completed_matches + 1:
        raise RuntimeError(
            f"Expected {completed_matches + 1} summary rows, "
            f"generated {len(snapshot_summary)}."
        )

    final_snapshot = history[
        history["checkpoint_index"]
        == history["checkpoint_index"].max()
    ]
    champion_probability = float(
        final_snapshot.loc[
            final_snapshot["team"] == app.ACTUAL_CHAMPION,
            "champion",
        ].iloc[0]
    )

    if champion_probability != 1.0:
        raise RuntimeError(
            "The final checkpoint should lock Spain at 100% champion "
            f"probability, but generated {champion_probability:.6f}."
        )

    write_csv_atomic(history, HISTORY_PATH)
    write_csv_atomic(snapshot_summary, SUMMARY_PATH)
    write_json_atomic(final_report, REPORT_PATH)

    print("\nPrediction archive rebuilt successfully.")
    print(f"  History rows: {len(history):,}")
    print(f"  Checkpoints: {len(snapshot_summary)}")
    print(f"  Teams per checkpoint: {len(app.INITIAL_TEAMS)}")
    print(f"  Final champion: {app.ACTUAL_CHAMPION}")
    print("\nGenerated files:")
    print(f"  {HISTORY_PATH.relative_to(ROOT)}")
    print(f"  {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"  {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
