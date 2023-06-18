import logging
import time

import numpy as np
import pandas as pd
import sqlalchemy
from scipy.stats import beta


# from datetime import timedelta



logger = logging.getLogger(__name__)


def get_rfecv_data(model: str, pair: str):
    """Get dataframe with feature selection stats from sqlite"""

    # Read trial + win data from sqlite
    connection_string = "sqlite:///litmus.sqlite"
    connection = sqlalchemy.create_engine(connection_string)
    sql = f"""
            SELECT feature_id, feature_rank, train_time, important_feature
            FROM rfecv_feature_selection
            WHERE model = '{model}'
            AND pair = '{pair}'"""

    try:
        data = pd.read_sql_query(sql=sqlalchemy.text(sql), con=connection.connect())
        return data
    except Exception as e:
        logger.info(f"Issue reading from SQL to exclude features {e}")
        return []


def get_rfecv_feature_to_exclude(model: str, pair: str, remove_weak_features_gt_rank: int,
                                 rerun_period_hours: float):
    """Get list of features to be excluded from training data"""

    data = get_rfecv_data(model, pair)

    if len(data) > 0:
        all_features = data["feature_id"].unique().tolist()
        last_run_timestamp = data["train_time"].max()
        rerun_timestamp = time.time() - rerun_period_hours * 60 * 60

        if last_run_timestamp < rerun_timestamp:
            # RFECV needs to be rerun / refreshed. Exclude features over threshold.
            logger.info("RFECV will be rerun")
            rfecv_rerun = True

            feature_rank_mean = data.groupby("feature_id")["feature_rank"].mean()
            features_to_keep = feature_rank_mean[
                feature_rank_mean <= remove_weak_features_gt_rank].index.tolist()

        else:
            # RFECV doesn't need to be rerun. Get best features from last run.
            logger.info("Skipping RFECV as it was run recently")
            rfecv_rerun = False

            latest_data = data[data["train_time"] == last_run_timestamp]
            features_to_keep = latest_data[
                latest_data["feature_rank"] == 1]["feature_id"].tolist()

    else:
        # No data in database. Rerun without excluding any features.
        logger.info("Need to run RFECV from scratch")
        rfecv_rerun = True
        all_features = []
        features_to_keep = []

    features_to_exclude = [f for f in all_features if f not in features_to_keep]

    return rfecv_rerun, features_to_exclude


def get_shap_rfecv_data(id: str, model: str, pair: str):
    """Get dataframe with feature selection stats from sqlite"""

    # Read trial + win data from sqlite
    connection_string = "sqlite:///litmus.sqlite"
    connection = sqlalchemy.create_engine(connection_string)
    sql = f"""
            SELECT feature_id, feature_rank, train_time
            FROM shap_rfecv_feature_selection
            WHERE id = '{id}'
            AND model = '{model}'
            AND pair = '{pair}'"""

    try:
        data = pd.read_sql_query(sql=sqlalchemy.text(sql), con=connection.connect())
        return data
    except Exception as e:
        logger.info(f"Issue reading from SQL to exclude features {e}")
        return []


def get_shap_rfecv_feature_to_exclude(
        id: str, model: str, pair: str, is_win_threshold: int, rerun_period_hours: float):
    """Get list of features to be excluded from training data"""

    data = get_shap_rfecv_data(id, model, pair)

    if len(data) > 0:
        all_features = data["feature_id"].unique().tolist()
        last_run_timestamp = data["train_time"].max()
        rerun_timestamp = time.time() - rerun_period_hours * 60 * 60

        if last_run_timestamp < rerun_timestamp:
            # RFECV needs to be rerun / refreshed. Exclude features over threshold.
            logger.info("Shap RFECV will be rerun")
            shap_rfecv_rerun = True

            data["is_win"] = data["feature_rank"] < is_win_threshold
            feature_rank_df = data.groupby("feature_id").agg(
                {"is_win": [("trials", "size"), ("wins", "sum")]}
            ).droplevel(level=0, axis=1).reset_index()

            feature_rank_df["beta_rand"] = feature_rank_df.apply(
                lambda row: beta(row["wins"] + 1, row["trials"] - row["wins"] + 1).rvs(), axis=1)
            # feature_rank_df["rand"] = np.random.random(size=len(feature_rank_df))
            # Beta(2,1) generates more proba with higher values
            feature_rank_df["rand"] = np.random.beta(a=2, b=1, size=len(feature_rank_df))

            features_to_keep = feature_rank_df[
                feature_rank_df["rand"] < feature_rank_df["beta_rand"]
            ]["feature_id"].tolist()

            latest_data = data[data["train_time"] == last_run_timestamp]
            last_features_to_keep = latest_data[
                latest_data["feature_rank"] == 1]["feature_id"].tolist()

            features_to_keep = features_to_keep + last_features_to_keep

            """feature_rank_df = feature_rank_df.sort_values(by="beta_rand", ascending=False)
            features_to_keep = feature_rank_df["feature_id"].iloc[:num_features_to_try].tolist()"""
            print(feature_rank_df.sort_values(by="wins", ascending=False).head(200))

        else:
            # RFECV doesn't need to be rerun. Get best features from last run.
            logger.info("Skipping Shap RFECV as it was run recently")
            shap_rfecv_rerun = False

            latest_data = data[data["train_time"] == last_run_timestamp]
            features_to_keep = latest_data[
                latest_data["feature_rank"] == 1]["feature_id"].tolist()

    else:
        # No data in database. Rerun without excluding any features.
        logger.info("Need to run Shap RFECV from scratch")
        shap_rfecv_rerun = True
        all_features = []
        features_to_keep = []

    features_to_exclude = [f for f in all_features if f not in features_to_keep]

    return shap_rfecv_rerun, features_to_exclude


def get_probatus_best_num_features(shap_report, best_method):
    if best_method == "aggressive_max_mean":
        shap_report["eval_metric"] = shap_report["val_metric_mean"]
        best_iteration_idx = shap_report["eval_metric"].argmax()
        best_num_features = shap_report["num_features"].iloc[best_iteration_idx]

    elif best_method == "conservative_max_mean":
        shap_report["eval_metric"] = (shap_report["val_metric_mean"]
                                      - shap_report["val_metric_std"] / 4.0)
        best_iteration_idx = shap_report["eval_metric"].argmax()
        best_num_features = shap_report["num_features"].iloc[best_iteration_idx]

    elif best_method == "min_ci_max_mean":
        shap_report["eval_metric"] = (shap_report["val_metric_mean"]
                                      - shap_report["val_metric_std"] / 4.0)
        best_iteration_idx = shap_report["eval_metric"].argmax()
        print(f"best_iteration_idx: {best_iteration_idx}")
        best_val_metric_threshold = shap_report["eval_metric"].iloc[best_iteration_idx]
        print(f"best_val_metric_threshold: {best_val_metric_threshold}")
        # drop iterations with val_metric below threshold
        shap_report = shap_report[shap_report["val_metric_mean"] > best_val_metric_threshold]
        # get iteration with smallest val_metric_std
        best_std_iteration_idx = shap_report["val_metric_std"].argmin()
        best_num_features = shap_report["num_features"].iloc[best_std_iteration_idx]
        print(f"best_num_features: {best_num_features}")

    return best_num_features


def get_probatus_best_feature_names(shap_refcv, shap_report, best_method):
    best_num_features = get_probatus_best_num_features(shap_report, best_method)
    best_feature_names = shap_refcv.get_reduced_features_set(best_num_features)
    print(shap_report)
    return best_feature_names


def get_probatus_feature_rank(shap_report, best_method):
    # Remove iterations below the best iteration
    best_iteration_num_features = get_probatus_best_num_features(
        shap_report, best_method=best_method)
    best_iteration_idx = shap_report[
        shap_report["num_features"] == best_iteration_num_features].index[0]
    shap_report = shap_report.iloc[:best_iteration_idx]

    # Add rank column and flip df
    shap_report["rank"] = np.flip(shap_report.index)
    shap_report = shap_report.iloc[::-1]

    results = []
    prev_row_features = []

    for i, row in shap_report.iterrows():

        current_row_features = row["features_set"]
        remaining_row_features = [v for v in current_row_features if v not in prev_row_features]
        prev_row_features = current_row_features

        for f in remaining_row_features:
            results.append([f, row["rank"]])

    results_df = pd.DataFrame(data=results, columns=["feature_id", "feature_rank"])

    return results_df
