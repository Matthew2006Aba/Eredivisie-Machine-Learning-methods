"""
Predicting Eredivisie match results with machine learning

Question: using only information available BEFORE kick-off (each team's recent
form and Elo rating), how well can different models predict home win / draw /
away win in the Dutch Eredivisie, and can they beat the bookmakers?

The script
  1. downloads Eredivisie results and match statistics (2017/18 - 2025/26),
  2. builds features from each team's last 5 matches (goals, shots, points...),
  3. tunes four models with grid search and time-series cross-validation:
       - logistic regression with an L2 penalty (ridge)
       - logistic regression with an L1 penalty (lasso)
       - PCA followed by logistic regression
       - random forest
  4. tests them on the last two seasons, which the models never saw,
  5. compares accuracy, log loss, stability per season and which features matter.

Run:  python football.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIRST_SEASON = 2017                 # 2017/18: first season with shot statistics
LAST_SEASON = 2025                  # 2025/26: last complete season
TEST_SEASONS = [2024, 2025]         # 2024/25 and 2025/26 are held out until the very end
FORM_WINDOW = 5                     # number of previous matches used for "form"
STATS = ["scored", "conceded", "shots", "shots_target", "corners", "points"]
URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"


# ---------------------------------------------------------------- data

def load_matches(cache="eredivisie.csv"):
    """All Eredivisie matches of the chosen seasons, sorted by date.

    The source file covers 38 leagues (~45 MB), so the first run keeps only the
    Eredivisie (division code N1) and saves it to a small CSV.
    """
    if not os.path.exists(cache):
        print("Downloading match data (first run only)...")
        all_matches = pd.read_csv(URL, low_memory=False)
        all_matches[all_matches["Division"] == "N1"].to_csv(cache, index=False)

    m = pd.read_csv(cache, parse_dates=["MatchDate"])
    m["Season"] = np.where(m["MatchDate"].dt.month >= 7, m["MatchDate"].dt.year, m["MatchDate"].dt.year - 1)
    m = m[m["Season"].between(FIRST_SEASON, LAST_SEASON)]
    return m.sort_values("MatchDate").reset_index(drop=True)


# ---------------------------------------------------------------- feature engineering

def team_form(matches, window=FORM_WINDOW):
    """Each team's average stats over its previous `window` matches.

    Every match appears twice here: once from the home team's view, once from
    the away team's. shift(1) makes sure a match never uses its own result.
    """
    home = pd.DataFrame({
        "match": matches.index, "team": matches["HomeTeam"], "date": matches["MatchDate"],
        "scored": matches["FTHome"], "conceded": matches["FTAway"],
        "shots": matches["HomeShots"], "shots_target": matches["HomeTarget"],
        "corners": matches["HomeCorners"],
        "points": matches["FTResult"].map({"H": 3, "D": 1, "A": 0}), "side": "home"})
    away = pd.DataFrame({
        "match": matches.index, "team": matches["AwayTeam"], "date": matches["MatchDate"],
        "scored": matches["FTAway"], "conceded": matches["FTHome"],
        "shots": matches["AwayShots"], "shots_target": matches["AwayTarget"],
        "corners": matches["AwayCorners"],
        "points": matches["FTResult"].map({"A": 3, "D": 1, "H": 0}), "side": "away"})
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "match"])

    long[STATS] = long.groupby("team")[STATS].transform(
        lambda x: x.shift(1).rolling(window, min_periods=window).mean())
    return long


def build_features(matches):
    """One row per match: home form, away form, the differences, and the Elo gap."""
    long = team_form(matches)
    home = long[long["side"] == "home"].set_index("match")[STATS].add_prefix("home_")
    away = long[long["side"] == "away"].set_index("match")[STATS].add_prefix("away_")
    X = home.join(away).sort_index()
    for s in STATS:
        X[f"diff_{s}"] = X[f"home_{s}"] - X[f"away_{s}"]
    X["elo_diff"] = matches["HomeElo"] - matches["AwayElo"]   # Elo rating known before the match
    return X


def train_test_split_by_season(matches, X):
    """Train on the older seasons, test on TEST_SEASONS. Rows without enough history are dropped."""
    data = X.join(matches[["Season", "FTResult", "OddHome", "OddDraw", "OddAway"]]).dropna(subset=list(X.columns))
    train = data[~data["Season"].isin(TEST_SEASONS)]
    test = data[data["Season"].isin(TEST_SEASONS)]
    features = list(X.columns)
    return train[features], train["FTResult"], test[features], test["FTResult"], test


# ---------------------------------------------------------------- models

def models():
    """The four models and the settings grid search tries for each."""
    return {
        "Logistic (L2 / ridge)": (
            make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=0, max_iter=5000)),
            {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}),
        "Logistic (L1 / lasso)": (
            make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=1, solver="saga", max_iter=5000)),
            {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}),
        "PCA + logistic": (
            make_pipeline(StandardScaler(), PCA(), LogisticRegression(max_iter=5000)),
            {"pca__n_components": [2, 4, 6, 10], "logisticregression__C": [0.01, 0.1, 1]}),
        "Random forest": (
            RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [10, 30, 100]}),
    }


def tune(model, grid, X_train, y_train):
    """Grid search with time-series cross-validation: every fold trains on
    earlier matches and validates on later ones, never the other way round."""
    search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=5), scoring="neg_log_loss")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------- evaluation

def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)
    return {"accuracy": accuracy_score(y_test, model.predict(X_test)),
            "log loss": log_loss(y_test, proba, labels=model.classes_)}


def baselines(y_train, test):
    """Simple benchmarks: always 'home win', and the bookmaker's odds."""
    y_test = test["FTResult"]
    # bookmaker: turn odds into probabilities (1/odds), rescaled to sum to 1
    implied = 1 / test[["OddAway", "OddDraw", "OddHome"]].to_numpy()
    implied = implied / implied.sum(axis=1, keepdims=True)
    bookie_pick = np.array(["A", "D", "H"])[implied.argmax(axis=1)]
    return {
        "Always 'home win'": {"accuracy": (y_test == "H").mean(), "log loss": np.nan},
        "Bookmaker odds (Bet365)": {"accuracy": (y_test == bookie_pick).mean(),
                                    "log loss": log_loss(y_test, implied, labels=["A", "D", "H"])},
    }


def interpretability(fitted, feature_names):
    """Which features matter: coefficients (home-win class) and forest importances."""
    out = pd.DataFrame(index=feature_names)
    for name in ["Logistic (L2 / ridge)", "Logistic (L1 / lasso)"]:
        lr = fitted[name][-1]
        home_row = list(lr.classes_).index("H")
        out[name] = lr.coef_[home_row]
    out["Random forest importance"] = fitted["Random forest"].feature_importances_
    return out.sort_values("Random forest importance", ascending=False)


# ---------------------------------------------------------------- main

def main():
    matches = load_matches()
    print(f"{len(matches)} Eredivisie matches, seasons {FIRST_SEASON}/{FIRST_SEASON + 1 - 2000} "
          f"to {LAST_SEASON}/{LAST_SEASON + 1 - 2000}")

    X = build_features(matches)
    X_train, y_train, X_test, y_test, test = train_test_split_by_season(matches, X)
    print(f"Training: {len(X_train)} matches   Test (last two seasons): {len(X_test)} matches")
    print("Result shares in training:", y_train.value_counts(normalize=True).round(3).to_dict())

    results, per_season, fitted = {}, {}, {}
    for name, (model, grid) in models().items():
        best, params = tune(model, grid, X_train, y_train)
        fitted[name] = best
        results[name] = evaluate(best, X_test, y_test)
        per_season[name] = {f"{s}/{s + 1 - 2000}": accuracy_score(y_test[test["Season"] == s],
                                                                  best.predict(X_test[test["Season"] == s]))
                            for s in TEST_SEASONS}
        print(f"  {name:24s} best settings: {params}")
    results.update(baselines(y_train, test))

    table = pd.DataFrame(results).T
    print("\nOut-of-sample results (test seasons)")
    print(table.round(3).to_string())
    print("\nAccuracy per test season (robustness)")
    print(pd.DataFrame(per_season).T.round(3).to_string())

    importance = interpretability(fitted, X.columns)
    lasso_zero = (importance["Logistic (L1 / lasso)"] == 0).sum()
    print(f"\nLasso set {lasso_zero} of {len(importance)} features to exactly zero")
    print("Most important features:")
    print(importance.head(8).round(3).to_string())

    table.to_csv("results.csv")
    importance.to_csv("feature_importance.csv")

    ax = table["accuracy"].sort_values().plot.barh(figsize=(8, 4), color="tab:orange")
    line = ax.axvline(1 / 3, color="grey", linestyle="--", label="random guess (33%)")
    ax.set_xlabel("Accuracy on the test seasons")
    ax.set_title("Predicting Eredivisie results: model comparison")
    ax.set_xlim(0, 0.75)
    ax.legend(handles=[line], loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    print("\nSaved results.csv, feature_importance.csv and model_comparison.png")


if __name__ == "__main__":
    main()
