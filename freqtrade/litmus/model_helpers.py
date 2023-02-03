# Helper functions for models
import logging
import time

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MergedModel:
    """Used for packaging multiple models together so that freqai can pickle the
    object for later usage

    Example usage:
        models = []
        models.append(model)
        model = MergedModel(models)"""

    def __init__(self, models):
        self.models = models
        self.classes_ = []
        for model in self.models:
            self.classes_.extend(list(model.classes_))

    def predict(self, X):
        results = []
        for model in self.models:
            results.append(model.predict(X))
        return np.hstack(results)

    def predict_proba(self, X):
        results = []
        for model in self.models:
            results.append(model.predict_proba(X))
        return np.hstack(results)


def exclude_weak_features(model: str, pair: str, loss_ratio_threshold: float,
                          chance_excluded: float, min_num_trials: int):
    """Identify weakest features from prior model feature selection routines
    and exclude these from future training

    :return list of column names that should be excluded for model + pair combo"""

    # Read trial + win data from sqlite
    connection_string = "sqlite:///litmus.sqlite"
    timestamp_in_past = time.time() - 10 * 24 * 60 * 60
    sql = f"""
        SELECT feature_id, importance
        FROM feature_importance_history
        WHERE model = '{model}'
        AND pair = '{pair}'
        AND train_time > '{timestamp_in_past}'"""

    try:
        data = pd.read_sql_query(sql=sql, con=connection_string)
    except Exception as e:
        logger.info(f"Issue reading from SQL to exclude features {e}")

    data["is_important"] = data["importance"] > 0
    sum_is_good = data.groupby("feature_id")["is_important"].sum()
    count_is_good = data.groupby("feature_id")["is_important"].size()
    summary_df = pd.concat(
        [sum_is_good, count_is_good], keys=["wins", "trials"], axis=1).reset_index()
    summary_df["loss_ratio"] = 1 - summary_df["wins"] / summary_df["trials"]

    # Generate random variate from distribution per feature based on prior inclusion / exclusion
    summary_df["random"] = np.random.random(size=len(summary_df))

    excluded = summary_df.loc[
        (summary_df["loss_ratio"] > loss_ratio_threshold) &
        (summary_df["random"] < chance_excluded) &
        (summary_df["trials"] > min_num_trials)]

    return excluded
