import logging
import time

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def get_unimportant_features(model: str, pair: str, pct_additional_features: float):
    """Get table with feature rank statistics to be used to keep and remove features"""

    # TODO: Make sure at least N trials complete before feature deemed unimportant

    # Read trial + win data from sqlite
    connection_string = "sqlite:///litmus.sqlite"
    timestamp_in_past = time.time() - 10 * 24 * 60 * 60
    sql = f"""
        SELECT feature_id, important_feature, train_time
        FROM feature_shuffle_selection
        WHERE model = '{model}'
        AND pair = '{pair}'
        AND train_time > '{timestamp_in_past}'"""

    try:
        data = pd.read_sql_query(sql=sql, con=connection_string)
    except Exception as e:
        logger.info(f"Issue reading from SQL to exclude features {e}")
        return []

    all_features = data["feature_id"].unique()

    # Get all selected features from last iteration
    last_iteration_time = data["train_time"].max()
    last_iteration_df = data[data["train_time"] == last_iteration_time]
    last_selected_features = last_iteration_df[
        last_iteration_df["important_feature"] == 1]["feature_id"].to_numpy()

    # Add X% more features ranked by selected ratio over past N iterations
    data = data[~data["feature_id"].isin(last_selected_features)]
    num_remaining_features = len(data["feature_id"].unique())

    sum_is_selected = data.groupby("feature_id")["important_feature"].sum()
    count_is_selected = data.groupby("feature_id")["important_feature"].size()
    summary_df = pd.concat(
        [sum_is_selected, count_is_selected],
        keys=["sum_is_selected", "count_is_selected"], axis=1).reset_index()

    summary_df["selected_ratio"] = (1 - summary_df["sum_is_selected"] /
                                    summary_df["count_is_selected"])
    summary_df = summary_df.sort_values(by="selected_ratio", ascending=False)
    num_additional_features = int(num_remaining_features * pct_additional_features)
    additional_features = summary_df["feature_id"][:num_additional_features].to_numpy()

    # Subtract important from all features to get unimportant features
    important_features = np.concatenate([last_selected_features, additional_features])
    unimportant_features = [c for c in all_features if c not in important_features]

    return unimportant_features


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


def get_rfecv_feature_importance(model: str, pair: str, exclude_rank_threshold: int,
                                 keep_rank_threshold: int, exclude_ratio_threshold: float,
                                 keep_ratio_threshold: float, chance_excluded: float,
                                 min_num_trials: int):
    """Get table with feature rank statistics to be used to keep and remove features"""

    # Read trial + win data from sqlite
    connection_string = "sqlite:///litmus.sqlite"
    timestamp_in_past = time.time() - 10 * 24 * 60 * 60
    sql = f"""
        SELECT feature_id, feature_rank
        FROM feature_rfecv_rank
        WHERE model = '{model}'
        AND pair = '{pair}'
        AND train_time > '{timestamp_in_past}'"""

    try:
        data = pd.read_sql_query(sql=sql, con=connection_string)
    except Exception as e:
        logger.info(f"Issue reading from SQL to exclude features {e}")

    data["is_unimportant"] = data["feature_rank"] > exclude_rank_threshold
    data["is_important"] = data["feature_rank"] <= keep_rank_threshold

    sum_is_unimportant = data.groupby("feature_id")["is_unimportant"].sum()
    count_is_unimportant = data.groupby("feature_id")["is_unimportant"].size()

    sum_is_important = data.groupby("feature_id")["is_important"].sum()
    count_is_important = data.groupby("feature_id")["is_important"].size()

    summary_df = pd.concat(
        [sum_is_unimportant, count_is_unimportant, sum_is_important, count_is_important],
        keys=["sum_exclude", "count_exclude", "sum_keep", "count_keep"], axis=1).reset_index()

    summary_df["exclude_ratio"] = 1 - summary_df["sum_exclude"] / summary_df["count_exclude"]
    summary_df["keep_ratio"] = 1 - summary_df["sum_keep"] / summary_df["count_keep"]

    # Generate random variate from distribution per feature based on prior inclusion / exclusion
    summary_df["random"] = np.random.random(size=len(summary_df))
    summary_df["confirm_exclude"] = (
            (summary_df["exclude_ratio"] >= exclude_ratio_threshold) &
            (summary_df["count_exclude"] > min_num_trials) &
            (summary_df["random"] < chance_excluded)
    )
    summary_df["confirm_keep"] = (
            (summary_df["keep_ratio"] >= keep_ratio_threshold) &
            (summary_df["count_keep"] > min_num_trials)
    )

    return summary_df
