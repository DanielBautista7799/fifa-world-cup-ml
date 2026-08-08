# FIFA World Cup ML Predictor

An end-to-end machine learning project that predicts international football match outcomes, simulates tournament results, and preserves how the model's World Cup probabilities changed throughout the completed tournament.

Live demo: https://fifa-world-cup-ml.streamlit.app/

---

## Final Project Status

The 2026 FIFA World Cup has ended, and this project is now published as a completed post-tournament prediction archive.

**Champion:** Spain  
**Runner-up:** Argentina  
**Third place:** England  
**Final:** Spain 1–0 Argentina after extra time

The deployed app now focuses on the completed tournament, saved prediction history, and forecast accountability. Normal page loads use committed CSV and JSON archive files instead of automatically rerunning thousands of simulations.

---

## Overview

This project is an end-to-end machine learning system for predicting FIFA World Cup outcomes.

The goal was not just to call a model from a library and get a result. The goal was to actually understand the full machine learning workflow:

* how match data becomes features
* how models learn from those features
* how probability predictions work
* how to evaluate models honestly
* how to simulate a tournament with Monte Carlo methods
* how to turn the final model into a live portfolio dashboard

The final app shows the official champion, the completed knockout path, every saved prediction checkpoint, probability timelines, historical match predictions, and temporary what-if scenarios.

The model is trained on historical international match data and uses engineered football features like Elo difference, recent form, rest days, tournament importance, head-to-head record, and attack/defense strength.

---

## Live Demo

The completed dashboard is available here:

[Open the FIFA World Cup ML Predictor](https://fifa-world-cup-ml.streamlit.app/)

The application now has six main sections:

1. **Dashboard** — champion, final result, key forecast findings, and probability timeline.
2. **Scenario Lab** — temporary historical what-if simulations that run only when requested.
3. **Past Predictions** — complete checkpoint history, comparison charts, and downloadable archive data.
4. **Match Probabilities** — historical match predictions from the information available at each checkpoint.
5. **Official Results** — the completed official knockout sequence.
6. **About the Model** — model selection, features, evaluation, design, and limitations.

The public app reads saved archive files during normal use. CPU-heavy simulations and model predictions only run after a visitor explicitly clicks an interactive action.

---

## What the Project Predicts

The project predicts match outcome probabilities:

```text
home win
draw
away win
```

For knockout matches, the model still predicts the normal three-class outcome, but the simulator removes the draw probability and normalizes the win probabilities so one team advances.

Example:

```text
Brazil win: 50%
Draw: 20%
Norway win: 30%
```

For a knockout simulation, the draw is removed:

```text
Brazil advances = 50 / (50 + 30)
Norway advances = 30 / (50 + 30)
```

This makes the model usable for tournament simulation while still keeping the original match model honest.

---

## Final Model

I compared three models:

| Model               | Accuracy | Log Loss |
| ------------------- | -------: | -------: |
| Gradient Boosting   | 0.603610 | 0.872078 |
| Random Forest       | 0.580559 | 0.897091 |
| Logistic Regression | 0.581168 | 0.899839 |

Gradient Boosting was selected because it had the best accuracy and the best log loss.

Accuracy matters, but log loss was especially important because this project depends on probability quality. A tournament simulator needs useful probabilities, not just hard winner guesses.

---

## Features Used

The final model uses these features:

```text
elo_difference
home_advantage
recent_win_rate_difference
recent_goals_for_difference
recent_goals_against_difference
recent_goal_difference_difference
rest_days_difference
streak_difference
is_world_cup
is_qualification
is_friendly
tournament_importance
head_to_head_difference
attack_rating_difference
defense_rating_difference
```

Most of these are difference features.

For example:

```text
elo_difference = home_team_elo - away_team_elo
```

This makes the model focus on the matchup instead of treating each team separately.

---

## Forecast Accountability and Prediction Archive

The project preserves what the model believed at every official checkpoint instead of only showing the final state after the tournament ended.

At the first saved checkpoint:

```text
Argentina champion probability: 31.10%
Spain champion probability: 21.06%
Spain rank: #2
```

Spain was not the original top projection, but the model assigned the eventual champion a meaningful chance from the beginning.

The permanent archive files are:

```text
data/app/prediction_history.csv
data/app/prediction_snapshot_summary.csv
data/app/final_model_report.json
```

These files preserve each team's quarterfinal, semifinal, final, and champion probability across every saved checkpoint. Because they are committed to the repository, the charts remain available without rerunning the simulations.

---

## Main Ideas Learned

This project covered the full ML pipeline from scratch to deployment.

### Python and Data Work

I practiced:

* pandas data cleaning
* working with dates
* filtering match rows
* creating new feature columns
* saving processed datasets
* loading saved models
* building repeatable notebook workflows

### Machine Learning Foundations

I worked through:

* linear regression
* logistic regression
* softmax classification
* loss functions
* gradient descent
* model evaluation
* probability prediction
* train/test splitting
* data leakage

The important part was understanding why each step exists instead of only trying to get a high score.

### Elo Rating System

I implemented and used Elo-style team ratings.

Elo helped represent team strength over time. The key idea is that ratings update after each match based on expected result versus actual result.

Upsets cause bigger rating changes than expected wins.

### Time-Safe Feature Engineering

One of the biggest lessons in this project was avoiding data leakage.

A prediction for a match can only use information that existed before that match.

That means features like recent form, Elo rating, rest days, and attack/defense strength must be calculated chronologically.

The basic rule was:

```text
calculate features before the match
then update team history after the match
```

This prevents the model from accidentally learning from the future.

### Model Selection

I trained and compared multiple models:

* Logistic Regression

* Random Forest

* Gradient Boosting

The final model was chosen using evaluation results instead of guessing.

### Monte Carlo Simulation

After creating match probabilities, I used Monte Carlo simulation to estimate tournament outcomes.

The app simulates the remaining tournament thousands of times. Each simulated tournament produces a champion. Repeating that process gives estimated probabilities for each team reaching each stage.

Example:

```text
If France wins 1,800 out of 10,000 simulations:
France champion probability = 18%
```

### Deployment

The final model was wrapped in a Streamlit dashboard so the project can be viewed live.

The deployed version shows:

* projected champion
* top 3 champion odds
* champion probability chart
* final appearance chart
* match probability breakdowns
* temporary scenario testing
* explanation of the model and limitations

---

## Dashboard Features

### Dashboard

The main page presents the completed tournament and the most important model findings:

* Spain as world champion
* Spain 1–0 Argentina after extra time
* Argentina as runner-up
* England as the third-place team
* initial projected champion
* Spain's initial probability and rank
* champion-probability timeline
* forecast-accountability summary

### Scenario Lab

The Scenario Lab allows temporary historical what-if scenarios. A visitor can start from a saved checkpoint, change a future result, and manually request a new Monte Carlo simulation.

Nothing runs automatically, and visitor scenarios do not modify the official archive.

### Past Predictions

This tab contains:

* champion-probability timeline
* team comparison controls
* checkpoint explorer
* saved probability tables
* downloadable prediction-history files

### Match Probabilities

This tab allows visitors to choose a historical checkpoint and calculate the model's probabilities for a match that was unresolved at that time.

It shows:

* home-win probability
* draw probability
* away-win probability
* adjusted knockout advancement probabilities

The saved model is loaded only when a visitor requests a calculation.

### Official Results

This tab shows the completed official match sequence used to build the archive.

### About the Model

This section explains the selected model, evaluation results, engineered features, simulation design, archive workflow, and limitations.

---

## Post-Tournament Runtime Design

The tournament is over, so the deployed website no longer reruns the complete tournament simulation whenever it starts, refreshes, or wakes from inactivity.

Normal page loads read committed data files for:

1. the champion and final standings
2. official match results
3. historical prediction checkpoints
4. probability timelines
5. forecast-accountability metrics
6. the final model report

Interactive simulations only run after a visitor presses the Scenario Lab simulation button. Historical match predictions only run after a visitor presses the calculation button.

This keeps the completed project fast and avoids unnecessary CPU usage while preserving the interactive ML demonstration.

---

## Repository Structure

```text
fifa-world-cup-ml/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── app/
│   │   ├── official_results.json
│   │   ├── tournament_summary.json
│   │   ├── prediction_history.csv
│   │   ├── prediction_snapshot_summary.csv
│   │   └── final_model_report.json
│   └── processed/
│       └── matches_with_advanced_features.csv
├── models/
│   ├── best_match_prediction_model.pkl
│   └── model_features.pkl
├── notebooks/
│   ├── 01_python_fundamentals.ipynb
│   ├── 02_linear_regression_from_scratch.ipynb
│   ├── 03_logistic_regression_from_scratch.ipynb
│   ├── 04_elo_ratings.ipynb
│   ├── 05_data_cleaning.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_model_training.ipynb
│   ├── 08_advanced_feature_engineering.ipynb
│   ├── 09_model_selection.ipynb
│   └── 10_live_world_cup_prediction.ipynb
├── scripts/
│   └── build_prediction_history.py
├── tests/
│   └── test_post_tournament_archive.py
└── .streamlit/
    └── config.toml
```

---

## Notebook Progression

### Notebook 01 — Python Fundamentals

Reviewed the Python concepts needed for the project:

* functions
* dictionaries
* loops
* lists
* basic testing ideas
* match data representation

### Notebook 02 — Linear Regression From Scratch

Built the foundation for understanding how machine learning models learn.

Covered:

* weights
* bias
* prediction
* mean squared error
* gradients
* gradient descent

### Notebook 03 — Logistic Regression From Scratch

Moved from regression to classification.

Covered:

* binary classification
* multiclass classification
* softmax
* cross-entropy
* probability prediction

### Notebook 04 — Elo Ratings

Built a team strength system.

Covered:

* expected result
* actual result
* rating updates
* upset handling
* using Elo as a feature

### Notebook 05 — Data Cleaning

Prepared the match dataset.

Covered:

* parsing dates
* cleaning columns
* creating result targets
* sorting chronologically
* preparing data for modeling

### Notebook 06 — Feature Engineering

Created the first serious football features.

Covered:

* home advantage
* Elo difference
* recent form
* rest days
* tournament type
* match importance

### Notebook 07 — Model Training

Trained the first machine learning models.

Covered:

* train/test split
* model fitting
* prediction
* probability outputs
* basic evaluation

### Notebook 08 — Advanced Feature Engineering

Improved the modeling table with stronger features.

Covered:

* attack ratings
* defense ratings
* streaks
* head-to-head history
* better difference features

### Notebook 09 — Model Selection

Compared models and chose the final one.

Models compared:

* Logistic Regression
* Random Forest
* Gradient Boosting

Gradient Boosting won and was saved as the final model.

### Notebook 10 — Live World Cup Prediction

Used the trained model to simulate the current live World Cup bracket.

Covered:

* loading the saved model
* building live match features
* predicting future matches
* handling knockout draws
* running 10,000 simulations
* saving final prediction results

### Final Streamlit Application

Turned the model into a completed portfolio application.

Covered:

* Streamlit deployment
* saved official results
* permanent prediction checkpoints
* forecast accountability
* historical probability charts
* manual what-if simulations
* post-tournament archive mode

---

## Running the Project Locally

Clone the repo:

```bash
git clone https://github.com/DanielBautista7799/fifa-world-cup-ml.git
cd fifa-world-cup-ml
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

The app should open at:

```text
http://localhost:8501
```

---

## Requirements

The deployed dashboard uses:

```text
joblib
matplotlib
numpy
pandas
plotly
scikit-learn
streamlit
```

The notebooks also use Jupyter for exploration.

---

## Rebuilding the Prediction Archive

The committed archive is used during normal website visits and does not need to be rebuilt.

To regenerate all checkpoints from the saved model:

```bash
python scripts/build_prediction_history.py --simulations 10000
```

This recreates:

```text
data/app/prediction_history.csv
data/app/prediction_snapshot_summary.csv
data/app/final_model_report.json
```

After rebuilding, run the tests and inspect the local website before committing the files.

---

## Current Limitations

This project is not trying to be a perfect sports betting model.

Main limitations:

* no player-level data
* no injuries
* no starting lineups
* no betting market odds
* no travel/weather data
* no tactical matchup data
* no automatic live score API
* knockout draws are handled with a simplified probability normalization
* the model updates live features, but does not retrain after every match

The point of the project is to show a complete machine learning workflow and an understandable prediction system.

---

## Possible Future Improvements

The current version is considered complete. Optional future extensions could include:

* probability calibration
* a Poisson goal model
* scoreline probability predictions
* player availability and injury features
* automatic official-score ingestion
* a richer bracket visualization
* uncertainty intervals
* historical World Cup backtesting
* a static archive mirror on the portfolio website

---

## What This Project Shows

This project demonstrates:

* Python data analysis
* feature engineering
* time-safe machine learning
* model evaluation
* probability prediction
* custom Elo-style ratings
* Monte Carlo simulation
* model persistence with joblib
* Streamlit deployment
* turning notebooks into a live portfolio app

The biggest learning point was that machine learning is not just calling `.fit()`.

The hard part is deciding what information the model is allowed to know, building features correctly, evaluating honestly, and turning the result into something useful.

---

## Resume Bullet

Built and deployed an end-to-end FIFA World Cup machine learning system using Python, pandas, scikit-learn, custom Elo-style ratings, time-safe feature engineering, Gradient Boosting, and Monte Carlo simulation; preserved 14 historical prediction checkpoints in an interactive post-tournament Streamlit archive.

---

## Final Project Summary

This project started as a learning exercise and became a complete machine learning portfolio project.

I built the workflow from historical match data through feature engineering, model training, evaluation, tournament simulation, live updating, deployment, and final archival.

The finished product does not pretend that football is perfectly predictable. Instead, it records what the model believed at each stage and compares those forecasts with what actually happened.

```text
Use historical information to estimate probabilities,
preserve those estimates honestly,
and evaluate them after the real tournament is complete.
```
