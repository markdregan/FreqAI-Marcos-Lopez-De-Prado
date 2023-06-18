import numpy as np
import pandas as pd


def get_predictions_with_returns(estimator, X, returns_df, target):
    X_copy = X.copy()
    class_idx = estimator.classes_.tolist().index(target)

    # Add target prediction probabilities to df
    X_copy["pred_proba"] = np.float64(estimator.predict_proba(X_copy)[:, class_idx])

    # Add returns to df
    if "short" in target:
        return_col = "returns_short"
    elif "long" in target:
        return_col = "returns_long"
    else:
        return -999

    X_copy["returns"] = returns_df.loc[:, return_col]
    X_copy = X_copy.sort_values(by="pred_proba", ascending=False)
    X_copy["cumprod_returns"] = X_copy["returns"].cumprod()

    return X_copy


def log_returns_description(estimator, X, y, returns_df, target):
    if target in estimator.classes_:

        # Get df with proba and returns
        df = get_predictions_with_returns(estimator, X, returns_df, target)

        # Print trade and cumprod returns descriptions
        bins = np.linspace(start=0, stop=1, num=11)
        stats = df[["cumprod_returns", "returns"]].groupby(
            pd.cut(df["pred_proba"], bins=bins)).agg(["mean", "min", "max", "count"])
        print(f"Target: {target}")
        print(stats)

        return -999
    else:
        return -999


def log_raw_predictions_description(estimator, X, y, returns_df, target):
    if target in estimator.classes_:

        # Get df with proba and returns
        pred_df = get_predictions_with_returns(estimator, X, returns_df, target)
        df = pd.concat([pred_df, y], axis=1)

        print(f"Target: {target}")
        print(df.head(500))

        return -999
    else:
        return -999


def get_threshold_max_cumprod_returns(estimator, X, y, returns_df, target):
    if target in estimator.classes_:

        # Get df with proba and returns
        df = get_predictions_with_returns(estimator, X, returns_df, target)

        # Print threshold and max value
        max_cumprod_returns_idx = df["cumprod_returns"].argmax()
        max_cumprod_returns_threshold = df.iloc[max_cumprod_returns_idx]["pred_proba"]

        return max_cumprod_returns_threshold
    else:
        return -999


def get_threshold_max_cumprod_returns_monte_carlo(
        estimator, X, y, returns_df, target, num_mc_iterations, mc_frac):

    if target in estimator.classes_:
        mc_results = []

        for _ in np.arange(num_mc_iterations):

            # Get df with proba and returns
            df = get_predictions_with_returns(estimator, X, returns_df, target)
            df = df.sample(
                frac=mc_frac, replace=True
            ).sort_values(
                by="pred_proba", ascending=False)

            # Print threshold and max value
            max_cumprod_returns_idx = df["cumprod_returns"].argmax()
            max_cumprod_returns_threshold = df.iloc[max_cumprod_returns_idx]["pred_proba"]

            mc_results.append(max_cumprod_returns_threshold)

        return np.median(mc_results)

    else:
        return -999


def get_value_max_cumprod_returns(estimator, X, y, returns_df, target):
    if target in estimator.classes_:

        # Get df with proba and returns
        df = get_predictions_with_returns(estimator, X, returns_df, target)

        # Print threshold and max value
        max_cumprod_returns_idx = df["cumprod_returns"].argmax()
        max_cumprod_returns_value = df.iloc[max_cumprod_returns_idx]["cumprod_returns"]

        return max_cumprod_returns_value
    else:
        return -999


def get_value_max_cumprod_returns_monte_carlo(
        estimator, X, y, returns_df, target, num_mc_iterations, mc_frac):

    if target in estimator.classes_:
        mc_results = []

        for _ in np.arange(num_mc_iterations):

            # Get df with proba and returns
            df = get_predictions_with_returns(estimator, X, returns_df, target)
            df = df.sample(
                frac=mc_frac, replace=True
            ).sort_values(
                by="pred_proba", ascending=False)

            # Print threshold and max value
            max_cumprod_returns_idx = df["cumprod_returns"].argmax()
            max_cumprod_returns_value = df.iloc[max_cumprod_returns_idx]["cumprod_returns"]

            mc_results.append(max_cumprod_returns_value)

        return np.mean(mc_results)
    else:
        return -999


def get_value_pct_y_true(estimator, X, y, target):
    if target in estimator.classes_:

        num_y_true = np.where(y == target, 1, 0).sum()
        denom = y.count()
        pct_y_true = np.divide(num_y_true, denom)

        return pct_y_true
    else:
        return -999
