"""Small checks that the features and the split are honest.  Run:  pytest"""
import numpy as np
import pandas as pd

from football import FORM_WINDOW, build_features, train_test_split_by_season


def fake_league(n_rounds=12, seed=0):
    """Four teams playing each other round after round, with random results."""
    rng = np.random.default_rng(seed)
    teams = ["Ajax", "PSV Eindhoven", "Feyenoord", "Twente"]
    rows = []
    date = pd.Timestamp("2022-08-06")
    for r in range(n_rounds):
        pairs = [(teams[0], teams[1 + r % 3]), (teams[1 + (r + 1) % 3], teams[1 + (r + 2) % 3])]
        for home, away in pairs:
            hg, ag = rng.integers(0, 4, 2)
            rows.append({"MatchDate": date, "HomeTeam": home, "AwayTeam": away,
                         "FTHome": hg, "FTAway": ag, "FTResult": "H" if hg > ag else "A" if ag > hg else "D",
                         "HomeShots": rng.integers(5, 20), "AwayShots": rng.integers(5, 20),
                         "HomeTarget": rng.integers(1, 8), "AwayTarget": rng.integers(1, 8),
                         "HomeCorners": rng.integers(0, 10), "AwayCorners": rng.integers(0, 10),
                         "HomeElo": 1500.0, "AwayElo": 1500.0,
                         "OddHome": 2.0, "OddDraw": 3.5, "OddAway": 3.5,
                         "Season": 2022 if r < n_rounds // 2 else 2024})
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_features_never_use_the_match_itself_or_later_matches():
    matches = fake_league()
    X = build_features(matches)
    changed = matches.copy()
    last = changed.index[-1]
    changed.loc[last, ["FTHome", "FTAway", "FTResult", "HomeShots"]] = [9, 0, "H", 40]
    X_changed = build_features(changed)
    pd.testing.assert_frame_equal(X, X_changed)   # the last match's own result is never a feature


def test_first_matches_have_no_form_yet():
    matches = fake_league()
    X = build_features(matches)
    # Ajax plays every round, so its first FORM_WINDOW home matches have no form history
    assert X["home_points"].iloc[:FORM_WINDOW].isna().all()


def test_training_data_comes_before_test_data():
    matches = fake_league()
    X = build_features(matches)
    X_train, y_train, X_test, y_test, test = train_test_split_by_season(matches, X)
    assert matches.loc[X_train.index, "MatchDate"].max() < matches.loc[X_test.index, "MatchDate"].min()


def test_form_is_the_average_of_the_previous_matches():
    matches = fake_league()
    X = build_features(matches)
    ajax_home = matches[matches["HomeTeam"] == "Ajax"]
    idx = ajax_home.index[FORM_WINDOW]           # Ajax's 6th match (always at home)
    previous = matches.loc[: idx - 1]
    ajax_prev = previous[(previous["HomeTeam"] == "Ajax") | (previous["AwayTeam"] == "Ajax")].tail(FORM_WINDOW)
    goals = np.where(ajax_prev["HomeTeam"] == "Ajax", ajax_prev["FTHome"], ajax_prev["FTAway"])
    assert np.isclose(X.loc[idx, "home_scored"], goals.mean())
