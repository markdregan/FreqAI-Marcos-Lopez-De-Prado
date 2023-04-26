import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


# Multiclass Methods

def get_mc_describe_returns(estimator, X, y, returns_df, title):
    classes_to_describe = [c for c in estimator.classes_ if c not in ["loss", "not_minmax_slow"]]
    for idx, c in enumerate(classes_to_describe):
        X_copy = X.copy()
        X_copy["pred_proba"] = estimator.predict_proba(X_copy)[:, idx]

        if "short" in c:
            return_col = "returns_short"
        else:
            return_col = "returns_long"

        returns = returns_df.loc[X_copy.index, return_col]
        bins = np.linspace(start=0, stop=1, num=11)
        stats = returns.groupby(pd.cut(X_copy["pred_proba"], bins=bins)).agg("describe")
        print(f"Trade Returns by Threshold: {title} ({c})")
        print(stats)

    return 0


def get_mc_describe_cumprod_returns(estimator, X, y, returns_df, title):
    classes_to_describe = [c for c in estimator.classes_ if c not in ["loss", "not_minmax_slow"]]
    for idx, c in enumerate(classes_to_describe):
        X_copy = X.copy()
        X_copy["pred_proba"] = estimator.predict_proba(X_copy)[:, idx]
        X_copy = X_copy.sort_values(by="pred_proba", ascending=False)

        if "short" in c:
            return_col = "returns_short"
        else:
            return_col = "returns_long"

        returns = returns_df.loc[X_copy.index, return_col]
        X_copy["cumprod_returns"] = returns.cumprod()

        bins = np.linspace(start=0, stop=1, num=11)
        stats = X_copy["cumprod_returns"].groupby(
            pd.cut(X_copy["pred_proba"], bins=bins)).agg("describe")
        print(f"Cumulative Product Returns by Threshold: {title} ({c})")
        print(stats)

    return 0


def get_mc_threshold_max_cumprod_returns(estimator, X, y, returns_df, target):
    classes_to_describe = [c for c in estimator.classes_]
    for idx, c in enumerate(classes_to_describe):

        if target == c:
            X_copy = X.copy()
            X_copy["pred_proba"] = estimator.predict_proba(X_copy)[:, idx]
            X_copy = X_copy.sort_values(by="pred_proba", ascending=False)

            if "short" in c:
                return_col = "returns_short"
            else:
                return_col = "returns_long"

            returns = returns_df.loc[X_copy.index, return_col]
            X_copy["cumprod_returns"] = returns.cumprod()

            max_cumprod_returns_idx = X_copy["cumprod_returns"].argmax()
            max_cumprod_returns_threshold = X_copy.iloc[max_cumprod_returns_idx]["pred_proba"]

            return max_cumprod_returns_threshold
        else:
            # Loop through remaining classes
            continue

    return -999


"""TODO
- Get threshold for max cum prod returns - allowing class to be specified
- Add trade cost to long and short situations appropriately
- Change trade returns to TBM"""


def get_mc_describe_precision_recall(estimator, X, y, title):
    classes_to_describe = [c for c in estimator.classes_ if c not in ["loss", "not_minmax_slow"]]
    for idx, c in enumerate(classes_to_describe):
        precision, recall, thresholds = precision_recall_curve(
            y, estimator.predict_proba(X)[:, idx], pos_label=c)

        pr_summary = np.column_stack([precision, recall, np.append(thresholds, [1])])
        pr_summary_df = pd.DataFrame(pr_summary, columns=["precision", "recall", "thresholds"])
        bins = np.linspace(start=0, stop=1, num=11)

        pr_agg_df = pr_summary_df[["precision", "recall"]].groupby(
            pd.cut(pr_summary_df["thresholds"], bins=bins)).mean()
        print(f"Precision Recall by Threshold: {title} ({c})")
        print(pr_agg_df)

    return 0


# Binary Classification Methods
def get_precision_at_threshold(estimator, X, y, desired_threshold):
    pos_label = np.sort(y.unique())[0]
    precision, recall, thresholds = precision_recall_curve(
        y, estimator.predict_proba(X)[:, 0], pos_label=pos_label)

    desired_threshold_idx = np.argmax(thresholds >= desired_threshold)

    return precision[desired_threshold_idx]


def get_recall_at_threshold(estimator, X, y, desired_threshold):
    pos_label = np.sort(y.unique())[0]
    precision, recall, thresholds = precision_recall_curve(
        y, estimator.predict_proba(X)[:, 0], pos_label=pos_label)

    desired_threshold_idx = np.argmax(thresholds >= desired_threshold)

    return recall[desired_threshold_idx]


def get_count(estimator, X, y, lower, upper):
    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    idx = (X_copy["pred_proba"] >= lower) & (X_copy["pred_proba"] <= upper)
    count = np.sum(idx)

    return count


def get_mean_returns(estimator, X, y, returns, lower, upper):
    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    returns = returns.loc[X_copy.index]
    idx = (X_copy["pred_proba"] >= lower) & (X_copy["pred_proba"] <= upper)
    return returns.loc[idx].mean()


def get_prod_returns(estimator, X, y, returns, lower, upper):
    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    returns = returns.loc[X_copy.index]
    idx = (X_copy["pred_proba"] >= lower) & (X_copy["pred_proba"] <= upper)
    return returns.loc[idx].product()


def get_threshold_max_cumprod_returns(estimator, X, y, returns):
    pos_label = np.sort(y.unique())[0]
    is_long = "_long" in pos_label

    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    returns = returns.loc[X_copy.index]
    X_copy = X_copy.sort_values(by="pred_proba", ascending=False)
    X_copy["cumprod_returns"] = returns.cumprod()
    if is_long:
        max_cumprod_returns_idx = X_copy["cumprod_returns"].argmax()
    else:
        max_cumprod_returns_idx = X_copy["cumprod_returns"].argmin()

    return X_copy.iloc[max_cumprod_returns_idx]["pred_proba"]


def get_value_max_cumprod_returns(estimator, X, y, returns):
    pos_label = np.sort(y.unique())[0]
    is_long = "_long" in pos_label

    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    returns = returns.loc[X_copy.index]
    X_copy = X_copy.sort_values(by="pred_proba", ascending=False)
    X_copy["cumprod_returns"] = returns.cumprod()
    if is_long:
        max_cumprod_returns_idx = X_copy["cumprod_returns"].argmax()
    else:
        max_cumprod_returns_idx = X_copy["cumprod_returns"].argmin()

    return X_copy.iloc[max_cumprod_returns_idx]["cumprod_returns"]


def get_precision_max_cumprod_returns(estimator, X, y, returns):
    threshold = get_threshold_max_cumprod_returns(estimator, X, y, returns)
    precision = get_precision_at_threshold(estimator, X, y, threshold)

    return precision


def get_recall_max_cumprod_returns(estimator, X, y, returns):
    threshold = get_threshold_max_cumprod_returns(estimator, X, y, returns)
    recall = get_recall_at_threshold(estimator, X, y, threshold)

    return recall


def get_describe_returns(estimator, X, y, returns, title):
    pos_label = np.sort(y.unique())[0]
    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    returns = returns.loc[X_copy.index]
    bins = np.linspace(start=0, stop=1, num=11)
    stats = returns.groupby(pd.cut(X_copy["pred_proba"], bins=bins)).agg("describe")
    print(f"Trade Returns by Threshold: {title} ({pos_label})")
    print(stats)
    return 0


def get_describe_precision_recall(estimator, X, y, title):
    pos_label = np.sort(y.unique())[0]
    precision, recall, thresholds = precision_recall_curve(
        y, estimator.predict_proba(X)[:, 0], pos_label=pos_label)

    pr_summary = np.column_stack([precision, recall, np.append(thresholds, [1])])
    pr_summary_df = pd.DataFrame(pr_summary, columns=["precision", "recall", "thresholds"])
    bins = np.linspace(start=0, stop=1, num=11)

    pr_agg_df = pr_summary_df[["precision", "recall"]].groupby(
        pd.cut(pr_summary_df["thresholds"], bins=bins)).agg(["mean", "median", "std"])
    print(f"Precision Recall by Threshold: {title} ({pos_label})")
    print(pr_agg_df)

    return 0


def get_describe_cumprod_returns(estimator, X, y, returns, title):
    pos_label = np.sort(y.unique())[0]
    X_copy = X.copy()
    X_copy["pred_proba"] = estimator.predict_proba(X)[:, 0]
    X_copy = X_copy.sort_values(by="pred_proba", ascending=False)

    returns = returns.loc[X_copy.index]
    X_copy["cumprod_returns"] = returns.cumprod()

    bins = np.linspace(start=0, stop=1, num=11)
    stats = X_copy["cumprod_returns"].groupby(
        pd.cut(X_copy["pred_proba"], bins=bins)).agg("describe")
    print(f"Cumulative Product Returns by Threshold: {title} ({pos_label})")
    print(stats)
    return 0
