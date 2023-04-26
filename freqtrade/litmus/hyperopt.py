# Helper functions for Optuna

import warnings

import optuna
from catboost import CatBoostClassifier
from optuna.exceptions import ExperimentalWarning
from sklearn.metrics import log_loss


# Suppress warning logs from Optuna
warnings.filterwarnings("ignore", category=ExperimentalWarning)

def get_study_name(pair, target):

    return pair + "-" + target


def get_best_hyperopt_params(study_name, storage_name):

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )
    return study.best_params


def get_ho_objective(trial: optuna.Trial, X_train, y_train, X_test, y_test) -> float:

    ho_params = {
        "objective": "Logloss",
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        "od_wait": trial.suggest_int("od_wait", 10, 50),
        "eval_metric": "Logloss"
    }

    if ho_params["bootstrap_type"] == "Bayesian":
        ho_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif ho_params["bootstrap_type"] == "Bernoulli":
        ho_params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    clf = CatBoostClassifier(**ho_params)

    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "Logloss")

    clf.fit(
        X=X_train, y=y_train, eval_set=[(X_test, y_test)],
        verbose=0, early_stopping_rounds=20,
        callbacks=[pruning_callback]
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    pred_proba = clf.predict_proba(X_test)[:, 0]

    return -log_loss(y_test, pred_proba)
