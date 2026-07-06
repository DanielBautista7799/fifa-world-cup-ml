# FIFA World Cup ML Predictor

A machine learning project that predicts international football match outcomes and simulates the remaining FIFA World Cup bracket.

Live demo: https://fifa-world-cup-ml.streamlit.app/

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

The final app lets users view the current projected champion, see top tournament probabilities, inspect individual match predictions, and test hypothetical match results in a temporary scenario lab.

The model is trained on historical international match data and uses engineered football features like Elo difference, recent form, rest days, tournament importance, head-to-head record, and attack/defense strength.

---

## Live Demo

The deployed dashboard is here:

[Open the FIFA World Cup ML Predictor](https://fifa-world-cup-ml.streamlit.app/)

The app has two main modes:

1. **Official projection**
   This uses the current committed tournament state and shows the model's official prediction.

2. **Scenario Lab**
   This lets users enter hypothetical match scores and instantly see how the probabilities change. These changes are temporary and reset when the page refreshes.

This was intentional. The public app is interactive, but visitors cannot permanently change the official results.

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

The main page shows the current official projection:

* projected champion
* top 3 champion probabilities
* locked official match results
* champion probability chart
* final appearance probability chart
* full probability table

### Scenario Lab

The Scenario Lab lets a user enter hypothetical match scores.

For example:

```text
Brazil 2 - 1 Norway
Portugal 1 - 0 Spain
```

The app then updates the bracket and reruns the simulation.

These scenario results are temporary. They do not save to the repo, database, or public app state.

This makes the dashboard safe for a public portfolio link while still letting people a public portfolio link while still letting people interact with the model.

### Past Predictions

This section is designed to show how official predictions change as real results are committed over time.

### Match Probabilities

This tab lets users inspect a single unresolved match and see:

* home win probability
* draw probability
* away win probability
* adjusted knockout advancement probability

### About the Model

This explains:

* what the model predicts
* which model was selected
* what features were used
* how the live updates work
* what the limitations are

---

## How Live Updating Works

The trained model itself is not retrained every time a score is entered.

Instead, the app updates the live tournament state.

When a score is entered in the Scenario Lab, the app:

1. records the temporary result for that user's session
2. advances the winning team
3. updates live Elo/form-style inputs
4. reruns the tournament simulation
5. shows the new probabilities

This is faster and cleaner than retraining the model every time.

For official real results, I update:

```text
data/app/official_results.json
```

Then I commit and redeploy the app.

That keeps the public app controlled while still letting it reflect real tournament progress.

---

## Repository Structure

```text
fifa-world-cup-ml/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── app/
│   │   └── official_results.json
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

### Notebook 11 — Live Dashboard

Turned the model into a deployed portfolio app.

Covered:

* Streamlit app
* live dashboard
* scenario testing
* temporary user state
* deployment

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

## Updating Official Results

The public dashboard uses this file for locked real-world results:

```text
data/app/official_results.json
```

Example:

```json
{
  "completed_matches": [
    {
      "match_id": "r16_2_left",
      "round": "round_of_16",
      "date": "2026-07-06",
      "home_team": "Portugal",
      "away_team": "Spain",
      "home_score": 1,
      "away_score": 0,
      "winner": "Portugal"
    }
  ]
}
```

After editing the file:

```bash
python -m streamlit run app.py
```

If the app loads correctly, commit and push:

```bash
git add data/app/official_results.json
git commit -m "Update World Cup dashboard after latest result"
git push
```

The deployed app will update after redeploy.

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

## What I Would Improve Next

If I keep expanding this project, the next improvements would be:

* add probability calibration
* add a Poisson goal model for scoreline prediction
* save final post-match team profiles directly instead of estimating from processed rows
* add a database for permanent public prediction snapshots
* add a cleaner bracket visualization
* add automatic score updates from a sports API
* add player availability and injury features
* build a more formal final model report

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

Built and deployed an interactive FIFA World Cup prediction dashboard using Python, pandas, scikit-learn, custom Elo-style team ratings, time-safe feature engineering, Gradient Boosting, and Monte Carlo simulation to update tournament probabilities as match results change.

---

## Final Project Summary

This project started as a learning exercise and turned into a full portfolio project.

I built the pipeline from raw match data to model training, selected the best model based on evaluation results, simulated the World Cup bracket thousands of times, and deployed a dashboard that lets users explore how different match results change the tournament probabilities.

The final output is not “this team will win.”

The final output is a probability-based system:

```text
Given the current tournament state and historical team features,
the model estimates each team's chance of reaching each round
and winning the World Cup.
```

That is the main idea of the project.
