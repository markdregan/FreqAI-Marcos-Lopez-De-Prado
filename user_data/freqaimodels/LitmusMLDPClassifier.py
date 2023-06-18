# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import sys
import tscv

from catboost import CatBoostClassifier
from feature_engine.selection import DropCorrelatedFeatures
# from fracdiff.sklearn import FracdiffStat
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.litmus.db_helpers import save_df_to_db
from freqtrade.litmus import feature_selection_helper as fs
from freqtrade.litmus import hyperopt
from freqtrade.litmus import model_diagnostics as md
from freqtrade.litmus.model_helpers import MergedModel
from functools import partial
from optuna.trial import TrialState
from pandas import DataFrame
from probatus.feature_elimination import EarlyStoppingShapRFECV
from probatus.utils import Scorer
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, f1_score, fbeta_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from telegram.helpers import escape_markdown
from time import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


class LitmusMLDPClassifier(IFreqaiModel):
    """
    Model that supports multiple classifiers trained separately with
    different features and model weights. Broadly aligns with guidance form marcos lopez de
    prado's book advances in financial machine learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define dict to store fracdiff params for faster transform
        self.custom_fds_info = dict()

    def train(
            self, unfiltered_dataframe: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_dataframe: Full dataframe for the current training period
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info("-------------------- Starting training " f"{pair} --------------------")

        # Extract df columns before normalization to be joined back below
        col_to_keep = [c for c in unfiltered_dataframe.columns if "!" in c]
        if len(col_to_keep) > 0:
            logger.info(f"Avoiding normalization for {len(col_to_keep)} columns: {col_to_keep}")
            df_keep = unfiltered_dataframe[col_to_keep]

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_dataframe,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_dataframe["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_dataframe["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to "
                    f"{end_date}--------------------")

        # split data into train/test data.
        data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions", 0) or not self.live:
            dk.fit_labels()

        # normalize all data based on train_dataset only
        data_dictionary = dk.normalize_data(data_dictionary)

        # Add un-normalized columns to data_dictionary
        if len(col_to_keep) > 0:
            train_idx = data_dictionary["train_features"].index
            test_idx = data_dictionary["test_features"].index
            data_dictionary["train_non_normalized"] = df_keep.loc[train_idx, :]
            data_dictionary["test_non_normalized"] = df_keep.loc[test_idx, :]

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f'Training model on {len(dk.data_dictionary["train_features"].columns)}' " features"
        )
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        model = self.fit(data_dictionary, dk)

        logger.info(f"-------------------- Done training {pair} --------------------")

        return model

    def predict(
            self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_dataframe)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_dataframe, dk.training_features_list, training_filter=False
        )

        filtered_dataframe = dk.normalize_data_from_metadata(filtered_dataframe)
        dk.data_dictionary["prediction_features"] = filtered_dataframe

        self.data_cleaning_predict(dk)

        # Apply fracdiff transform from prior fit
        # TODO: when bot restarted, "fds" does not exist. But this code below is called.
        """if self.freqai_info["feature_parameters"]["fracdiff"].get("enabled", False):

            X_df = dk.data_dictionary["prediction_features"]

            numeric_cols = X_df.select_dtypes("number").columns
            cols_to_fd = [c for c in numeric_cols if "%" in c and "_fd" not in c]
            new_col_names = [c + "_fd" for c in cols_to_fd]

            fds = self.custom_fds_info["fds"]
            X_df.loc[:, new_col_names] = fds.transform(X_df[cols_to_fd])
            X_df = X_df.drop(columns=cols_to_fd)

            dk.data_dictionary["prediction_features"] = X_df"""

        predictions = self.model.predict(dk.data_dictionary["prediction_features"])
        pred_df = DataFrame(predictions, columns=dk.label_list)

        predictions_prob = self.model.predict_proba(dk.data_dictionary["prediction_features"])
        pred_df_prob = DataFrame(predictions_prob, columns=self.model.classes_)

        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        return (pred_df, dk.do_predict)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:  # noqa: C901
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        start_time = time()

        # Object to wrap multiple models
        model_list = []
        model_features_list = []
        model_type_list = []

        for t in data_dictionary["train_labels"].columns:
            logger.info(f"Start training for target {t}")

            # Swap train and test data
            X_train = data_dictionary["test_features"]
            y_train = data_dictionary["test_labels"][t]

            time_scaler = MinMaxScaler(feature_range=(1, 10))
            time_weight_train = pd.DataFrame(
                time_scaler.fit_transform(
                    np.absolute(data_dictionary["test_weights"].reshape(-1, 1))),
                columns=["time_weight_train"], index=data_dictionary["test_labels"].index,
                dtype="float")

            utility_train = data_dictionary["test_non_normalized"]
            weight_scaler = MinMaxScaler(feature_range=(1, 10))
            weight_train = pd.DataFrame(
                weight_scaler.fit_transform(
                    np.absolute(utility_train.loc[:, ["!-trade_return"]])),
                columns=["weight_train"], index=utility_train.index, dtype="float").fillna(value=1)

            print(weight_train)
            print(time_weight_train)
            print(weight_train["weight_train"] + time_weight_train["time_weight_train"])
            # weight_train = weight_train + time_weight_train

            X_test = data_dictionary["train_features"]
            y_test = data_dictionary["train_labels"][t]
            time_weight_test = data_dictionary["train_weights"]  # noqa: F841
            utility_test = data_dictionary["train_non_normalized"]
            weight_test = pd.DataFrame(
                weight_scaler.transform(np.absolute(utility_test.loc[:, ["!-trade_return"]])),
                columns=["weight_test"], index=utility_test.index, dtype="float")

            # Summary of target class counts
            logger.info(f"Class counts for {t}")
            print(y_train.groupby(y_train).size())

            # Get best params from optuna
            if self.freqai_info["optuna"].get("enabled", False):
                ho_study_name = hyperopt.get_study_name(dk.pair, t)
                ho_storage_name = "sqlite:///LitmusOptuna.sqlite"
                try:
                    base_params = self.freqai_info["model_training_parameters"]
                    ho_best_params = hyperopt.get_best_hyperopt_params(
                        ho_study_name, ho_storage_name)
                    # ho_best_params.update(base_params)
                    logger.info(f"Optuna ON. Best params loaded: {ho_best_params}")
                except Exception:
                    ho_best_params = self.freqai_info["model_training_parameters"]
                    logger.info(f"Optuna ON. Default params loaded: {ho_best_params}")
            else:
                ho_best_params = self.freqai_info["model_training_parameters"]
                logger.info(f"Optuna OFF. Default params loaded: {ho_best_params}")

            # Apply fractional differentiation to features
            if self.freqai_info["feature_parameters"]["fracdiff"].get("enabled", False):
                fd_params = self.freqai_info["feature_parameters"]["fracdiff"]  # noqa: F841

                numeric_cols = X_train.select_dtypes("number").columns
                cols_to_fd = [c for c in numeric_cols if "%" in c and "_fd" not in c]
                new_col_names = [c + "_fd" for c in cols_to_fd]

                last_fit_timestamp = self.custom_fds_info.get("last_fit_timestamp", 0)
                if last_fit_timestamp + 6 * 24 * 60 * 60 > time():
                    logger.info("Applying fracdiff params from previous fit")
                    fds = self.custom_fds_info["fds"]
                    X_train.loc[:, new_col_names] = fds.transform(X_train[cols_to_fd])
                    logger.info("Finished applying fracdiff params from previous fit")

                else:
                    logger.info(f"Starting fracdiff procedure for {len(cols_to_fd)} columns")

                    """
                    # Note: Have not seen good impact from fracdiff. commenting out for now.
                    fds = FracdiffStat(
                        window=fd_params["window"], mode=fd_params["mode"],
                        precision=fd_params["precision"], upper=fd_params["upper"])

                    # Note: `row_stride` helps speed up fracdiff by subsampling
                    fds.fit(X_train.loc[::fd_params["row_stride"], cols_to_fd])
                    X_train.loc[:, new_col_names] = fds.transform(X_train[cols_to_fd])

                    # Save fds params for faster fit when needed
                    self.custom_fds_info["last_fit_timestamp"] = time()
                    self.custom_fds_info["fds"] = fds

                    logger.info("Finished fracdiff procedure")
                    pprint(dict(zip(cols_to_fd, fds.d_)))"""

                # Drop non fracdiff features
                X_train = X_train.drop(columns=cols_to_fd)

            # Shap RFECV: Drop weak features
            if self.freqai_info["feature_selection"]["shap_rfecv"].get("enabled", False):
                shap_rfecv_params = self.freqai_info["feature_selection"]["shap_rfecv"]

                self.shap_rfecv_rerun, features_to_exclude = fs.get_shap_rfecv_feature_to_exclude(
                    id=self.freqai_info["identifier"],
                    model=t,
                    pair=dk.pair,
                    is_win_threshold=shap_rfecv_params["is_win_threshold"],
                    rerun_period_hours=shap_rfecv_params["rerun_period_hours"]
                )

                logger.info(f"Size of X_train before removal: {len(X_train.columns)}")
                X_train = X_train.drop(columns=features_to_exclude, errors="ignore")
                logger.info(f"Size of X_train after removal: {len(X_train.columns)}")

            # Drop features using Greedy Correlated Selection
            if self.freqai_info["feature_selection"]["greedy_selection"].get("enabled", False):
                num_feat = len(X_train.columns)
                logger.info(f"Starting Greedy Feature Selection for {num_feat} features")

                # Get config params
                greedy_selection_params = self.freqai_info["feature_selection"]["greedy_selection"]

                greedy_selection = DropCorrelatedFeatures(
                    method="pearson",
                    threshold=greedy_selection_params["threshold"], missing_values="ignore",
                    confirm_variables=False)

                X_train = greedy_selection.fit_transform(X_train, y_train)
                X_test = greedy_selection.transform(X_test)

                num_remaining = len(X_train.columns)
                num_dropped = len(greedy_selection.features_to_drop_)
                logger.info(f"Dropping {num_dropped} correlated features using greedy. "
                            f"{num_remaining} remaining.")

            # Drop rows for meta-model to align with primary entries only
            if self.freqai_info["feature_selection"]["drop_rows_meta_model"].get("enabled", False):

                # Rows to keep
                train_row_keep_idx = y_train[y_train != "drop-row"].index
                test_row_keep_idx = y_test[y_test != "drop-row"].index

                # Keep specific rows
                X_train = X_train.loc[train_row_keep_idx, :]
                y_train = y_train.loc[train_row_keep_idx]
                weight_train = weight_train.loc[train_row_keep_idx, :]
                utility_train = utility_train.loc[train_row_keep_idx, :]

                X_test = X_test.loc[test_row_keep_idx, :]
                y_test = y_test.loc[test_row_keep_idx]
                weight_test = weight_test.loc[test_row_keep_idx, :]
                utility_test = utility_test.loc[test_row_keep_idx, :]

                logger.info(f"Meta-model prep: Dropped train rows and {len(y_train)} remaining.")
                logger.info(f"Meta-model prep: Dropped test rows and {len(y_test)} remaining.")

            # Shap RFECV
            self.shap_rfecv_enabled = self.freqai_info["feature_selection"]["shap_rfecv"].get(
                "enabled", False)
            if self.shap_rfecv_enabled and self.shap_rfecv_rerun:
                logger.info(f"Starting Shap RFECV for {len(X_train.columns)} features")
                shap_rfecv_params = self.freqai_info["feature_selection"]["shap_rfecv"]

                clf = CatBoostClassifier(**ho_best_params)
                # suppress_verbosity_params = {"verbose": False}
                # clf.set_params(**suppress_verbosity_params)

                cv_params = self.freqai_info["feature_selection"]["cv"]
                cv = tscv.GapKFold(
                    n_splits=cv_params["cv_n_splits"],
                    gap_before=cv_params["cv_gap_before"],
                    gap_after=cv_params["cv_gap_after"]
                )

                pos_label = [v for v in y_train.unique() if "win" in v][0]
                # TODO: Add sample_weight (Probatus not yet supporting)
                sample_weight = weight_train.to_numpy().ravel()

                """scorer = Scorer("f1", custom_scorer=make_scorer(
                    fbeta_score, beta=shap_rfecv_params["fbeta_coeff"],
                    pos_label=pos_label, average="binary"))"""

                # Use max_cum_prod_returns as it inherently captures sample_weights
                # Config: Model Diagnostic Scores
                trade_cost_pct = self.freqai_info["entry_parameters"].get("trade_cost_pct", 0)
                slippage_cost_pct = self.freqai_info["entry_parameters"].get("slippage_cost_pct", 0)
                returns_df = pd.DataFrame(index=utility_train.index)
                returns_df["returns_long"] = (
                        utility_train.loc[:, ["!-trade_return_long"]]
                        + 1.0 - trade_cost_pct - slippage_cost_pct
                )
                returns_df["returns_short"] = (
                        -1 * utility_train.loc[:, ["!-trade_return_short"]]
                        + 1.0 - trade_cost_pct - slippage_cost_pct
                )

                scorer = Scorer("max_cum_prod_returns", custom_scorer=partial(
                    md.get_value_max_cumprod_returns, returns_df=returns_df, target=pos_label))

                shap_refcv = EarlyStoppingShapRFECV(
                    clf=clf,
                    step=shap_rfecv_params["step"],
                    min_features_to_select=shap_rfecv_params["min_features_to_select"],
                    cv=cv,
                    scoring=scorer,
                    n_jobs=cv_params["cv_n_splits"],
                    verbose=shap_rfecv_params["verbose"],
                    early_stopping_rounds=shap_rfecv_params["early_stopping_rounds"]
                )

                shap_kwargs = {"feature_perturbation": "tree_path_dependent"}

                shap_report = shap_refcv.fit_compute(
                    X_train, y_train, sample_weight=sample_weight, **shap_kwargs)

                best_feature_names = fs.get_probatus_best_feature_names(
                    shap_refcv, shap_report, best_method="conservative_max_mean")

                X_train = X_train.loc[:, best_feature_names]

                # Store feature ranks to database
                feat_df = fs.get_probatus_feature_rank(
                    shap_report, best_method="conservative_max_mean")
                feat_df["train_time"] = time()
                feat_df["pair"] = dk.pair
                feat_df["model"] = t
                feat_df["id"] = self.freqai_info["identifier"]
                feat_df = feat_df.set_index(
                    keys=["id", "model", "train_time", "pair", "feature_id"])
                save_df_to_db(df=feat_df, table_name="shap_rfecv_feature_selection")

                num_features_selected = len(X_train.columns)
                logger.info(f"Shap RFECV complete selecting {num_features_selected} features")

            # Optuna hyper param optimization
            pct_triggered = self.freqai_info["optuna"]["pct_triggered"]
            random_selection = np.random.random() < pct_triggered
            if self.freqai_info["optuna"].get("enabled", False) & random_selection:
                logger.info("Starting Optuna Process")

                study = optuna.create_study(
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    direction="maximize",
                    study_name=ho_study_name,
                    storage=ho_storage_name,
                    load_if_exists=True
                )

                base_params = self.freqai_info["model_training_parameters"]
                clf = CatBoostClassifier(**base_params)

                param_distributions = {
                    "loss_function": optuna.distributions.CategoricalDistribution(
                        choices=("MultiClassOneVsAll", "Logloss")
                    ),
                    "iterations": optuna.distributions.IntDistribution(800, 1500, log=True),
                    "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.01, log=True),
                    "depth": optuna.distributions.IntDistribution(2, 8),
                    "l2_leaf_reg": optuna.distributions.FloatDistribution(
                        1e-8, 100.0, log=True),
                    "colsample_bylevel": optuna.distributions.FloatDistribution(
                        0.01, 0.1, log=True),
                    "bootstrap_type": optuna.distributions.CategoricalDistribution(
                        choices=("Bayesian", "Bernoulli", "MVS")
                    ),
                    "grow_policy": optuna.distributions.CategoricalDistribution(
                        choices=("SymmetricTree", "Depthwise", "Lossguide")),
                    "verbose": optuna.distributions.IntDistribution(200, 200)
                }

                cv_params = self.freqai_info["feature_selection"]["cv"]
                cv = tscv.GapKFold(
                    n_splits=cv_params["cv_n_splits"],
                    gap_before=cv_params["cv_gap_before"],
                    gap_after=cv_params["cv_gap_after"])

                # TODO: Might need to change score function for CV. Seems to impact perf.
                model = optuna.integration.OptunaSearchCV(
                    clf, param_distributions, cv=cv, enable_pruning=False,
                    study=study, n_trials=10, verbose=1, refit=False,
                    scoring=make_scorer(f1_score, greater_is_better=False, needs_proba=False))
                model.fit(X_train, y_train, sample_weight=weight_train.to_numpy().ravel())
                ho_best_params = model.best_params_
                # TODO: Add attrs - model.set_user_attr("dataset", "blobs")

                # Collect study trials to keep
                num_trials = study.trials[-1].number
                num_trials_to_keep = self.freqai_info["optuna"]["num_trials_to_keep"]

                if num_trials > num_trials_to_keep:
                    logger.info(f"Starting to reduce optuna study trials from {num_trials} to "
                                f"{num_trials_to_keep}")

                    trials_to_keep = [
                        optuna.trial.create_trial(
                            state=TrialState.COMPLETE,
                            params=t.params,
                            user_attrs=t.user_attrs,
                            system_attrs=t.system_attrs,
                            intermediate_values=t.intermediate_values,
                            distributions=t.distributions,
                            value=t.value
                        ) for t in study.trials if t.number > num_trials - num_trials_to_keep]

                    # Delete study before recreating
                    optuna.delete_study(study_name=ho_study_name, storage=ho_storage_name)

                    # Recreate study and add trials to keep
                    new_study = optuna.create_study(
                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                        direction="maximize",
                        study_name=ho_study_name,
                        storage=ho_storage_name,
                        load_if_exists=True)
                    new_study.add_trials(trials_to_keep)

                    logger.info("Successfully reduced optuna trials and saved to db")

            # Compute CV Model Performance Metrics
            model = CatBoostClassifier(**ho_best_params)

            cv_params = self.freqai_info["feature_selection"]["cv"]
            cv = tscv.GapKFold(
                n_splits=cv_params["cv_n_splits"],
                gap_before=cv_params["cv_gap_before"],
                gap_after=cv_params["cv_gap_after"])

            # Model diagnostics
            trade_cost_pct = self.freqai_info["entry_parameters"].get("trade_cost_pct", 0)
            slippage_cost_pct = self.freqai_info["entry_parameters"].get("slippage_cost_pct", 0)
            returns_df = pd.DataFrame(index=utility_train.index)
            returns_df["returns_long"] = (
                    utility_train.loc[:, ["!-trade_return_long"]]
                    + 1.0 - trade_cost_pct - slippage_cost_pct
            )
            returns_df["returns_short"] = (
                    -1 * utility_train.loc[:, ["!-trade_return_short"]]
                    + 1.0 - trade_cost_pct - slippage_cost_pct
            )

            pos_label = [v for v in y_train.unique() if "win" in v][0]
            shap_rfecv_params = self.freqai_info["feature_selection"]["shap_rfecv"]

            perf_mc_metrics = {
                "win_long_returns_description": partial(
                    md.log_returns_description,
                    returns_df=returns_df, target="win_long"),
                "win_long_threshold": partial(
                    md.get_threshold_max_cumprod_returns,
                    returns_df=returns_df, target="win_long"),
                "win_long_threshold_mc": partial(
                    md.get_threshold_max_cumprod_returns_monte_carlo,
                    returns_df=returns_df, target="win_long", num_mc_iterations=500, mc_frac=0.2),
                "win_long_value": partial(
                    md.get_value_max_cumprod_returns,
                    returns_df=returns_df, target="win_long"),
                "win_long_value_mc": partial(
                    md.get_value_max_cumprod_returns_monte_carlo,
                    returns_df=returns_df, target="win_long", num_mc_iterations=500, mc_frac=0.2),
                "win_long_pct_y_true": partial(
                    md.get_value_pct_y_true,
                    target="win_long"),

                "win_short_returns_description": partial(
                    md.log_returns_description,
                    returns_df=returns_df, target="win_short"),
                "win_short_threshold": partial(
                    md.get_threshold_max_cumprod_returns,
                    returns_df=returns_df, target="win_short"),
                "win_short_threshold_mc": partial(
                    md.get_threshold_max_cumprod_returns_monte_carlo,
                    returns_df=returns_df, target="win_short", num_mc_iterations=500, mc_frac=0.2),
                "win_short_value": partial(
                    md.get_value_max_cumprod_returns,
                    returns_df=returns_df, target="win_short"),
                "win_short_value_mc": partial(
                    md.get_value_max_cumprod_returns_monte_carlo,
                    returns_df=returns_df, target="win_short", num_mc_iterations=500, mc_frac=0.2),
                "win_short_pct_y_true": partial(
                    md.get_value_pct_y_true,
                    target="win_short"),

                "value_meta_f1_score": make_scorer(
                    f1_score, pos_label=pos_label, average="binary"),
                "value_f1_beta_score": make_scorer(
                    fbeta_score, beta=shap_rfecv_params["fbeta_coeff"],
                    pos_label=pos_label, average="binary"),
                "value_f1_macro_score": make_scorer(
                    f1_score, average="macro"),
                "value_auc_roc_score": make_scorer(
                    roc_auc_score, needs_proba=True),
                "value_neg_log_loss_score": make_scorer(
                    log_loss, needs_proba=True),
            }

            fit_params = {"sample_weight": weight_train.to_numpy().ravel()}
            perf_scores = cross_validate(
                estimator=model, X=X_train, y=y_train, scoring=perf_mc_metrics,
                cv=cv, n_jobs=cv_params["cv_n_splits"], fit_params=fit_params)

            # Telegram diagnostics
            telegram_dict = {}
            telegram_dict["pair"] = escape_markdown(dk.pair, version=2)
            telegram_dict["model"] = escape_markdown(t, version=2)

            # Long
            if -999 not in perf_scores["test_win_long_threshold"]:
                dk.data["extra_returns_per_train"][f"win_long_threshold_{t}"] = \
                    perf_scores["test_win_long_threshold"].mean()
                win_long_threshold = list(np.round(perf_scores["test_win_long_threshold"], 4))
                telegram_dict["win_long_threshold"] = ', '.join(str(x) for x in win_long_threshold)

            if -999 not in perf_scores["test_win_long_value"]:
                dk.data["extra_returns_per_train"][f"win_long_value_{t}"] = \
                    perf_scores["test_win_long_value"].mean()
                win_long_value = list(np.round(perf_scores["test_win_long_value"], 4))
                telegram_dict["win_long_value"] = ', '.join(str(x) for x in win_long_value)

            if -999 not in perf_scores["test_win_long_value_mc"]:
                win_long_value_mc = list(np.round(perf_scores["test_win_long_value_mc"], 4))
                telegram_dict["win_long_value_mc"] = ', '.join(
                    str(x) for x in win_long_value_mc)

            if -999 not in perf_scores["test_win_long_pct_y_true"]:
                win_long_pct_y_true = list(np.round(perf_scores["test_win_long_pct_y_true"], 4))
                telegram_dict["win_long_pct_y_true"] = ', '.join(
                    str(x) for x in win_long_pct_y_true)

            # Short
            if -999 not in perf_scores["test_win_short_threshold"]:
                dk.data["extra_returns_per_train"][f"win_short_threshold_{t}"] = \
                    perf_scores["test_win_short_threshold"].mean()
                win_short_threshold = list(np.round(perf_scores["test_win_short_threshold"], 4))
                telegram_dict["win_short_threshold"] = ', '.join(
                    str(x) for x in win_short_threshold)

            if -999 not in perf_scores["test_win_short_value"]:
                dk.data["extra_returns_per_train"][f"win_short_value_{t}"] = \
                    perf_scores["test_win_short_value"].mean()
                win_short_value = list(np.round(perf_scores["test_win_short_value"], 4))
                telegram_dict["win_short_value"] = ', '.join(
                    str(x) for x in win_short_value)

            if -999 not in perf_scores["test_win_short_value_mc"]:
                win_short_value_mc = list(np.round(perf_scores["test_win_short_value_mc"], 4))
                telegram_dict["win_short_value_mc"] = ', '.join(
                    str(x) for x in win_short_value_mc)

            if -999 not in perf_scores["test_win_short_pct_y_true"]:
                win_short_pct_y_true = list(np.round(perf_scores["test_win_short_pct_y_true"], 4))
                telegram_dict["win_short_pct_y_true"] = ', '.join(
                    str(x) for x in win_short_pct_y_true)

            # F1 Score
            dk.data["extra_returns_per_train"][f"value_meta_f1_score_{t}"] = \
                perf_scores["test_value_meta_f1_score"].mean()
            meta_f1_score = list(np.round(perf_scores["test_value_meta_f1_score"], 4))
            telegram_dict["f1_score"] = ', '.join(
                str(x) for x in meta_f1_score)

            # F1 Beta Score
            dk.data["extra_returns_per_train"][f"value_f1_beta_score_{t}"] = \
                perf_scores["test_value_f1_beta_score"].mean()
            f1_beta_score = list(np.round(perf_scores["test_value_f1_beta_score"], 4))
            telegram_dict["f1_beta_score"] = ', '.join(
                str(x) for x in f1_beta_score)

            # F1 Macro Score
            meta_f1_macro_score = list(np.round(perf_scores["test_value_f1_macro_score"], 4))
            telegram_dict["f1_macro_score"] = ', '.join(
                str(x) for x in meta_f1_macro_score)

            # AUC ROC Score
            auc_roc_score = list(np.round(perf_scores["test_value_auc_roc_score"], 4))
            telegram_dict["auc_roc_score"] = ', '.join(
                str(x) for x in auc_roc_score)

            # Neg Log Loss Score
            neg_log_loss_score = list(np.round(perf_scores["test_value_neg_log_loss_score"], 4))
            telegram_dict["neg_log_loss_score"] = ', '.join(
                str(x) for x in neg_log_loss_score)

            # Number Features
            num_features_selected = len(X_train.columns)
            dk.data["extra_returns_per_train"][
                f"num_features_selected_{t}"] = num_features_selected
            telegram_dict["num_features_selected"] = num_features_selected

            # Train time
            telegram_dict["fit_time"] = perf_scores["fit_time"].mean()

            # Print logs
            logger.info(f"CV model performance scores: {perf_scores}")

            # Log to telegram
            telegram_msg = []
            for k, v in telegram_dict.items():
                telegram_msg.append(f"*{k}:* {v}")
            self.data_provider.send_msg('\n'.join(telegram_msg))
            print(telegram_msg)

            # Train final model
            model = CatBoostClassifier(**ho_best_params)
            model.fit(X_train, y_train, sample_weight=weight_train.to_numpy().ravel())

            # Prepare model details for MergedModel class
            model_list.append(model)
            model_type_list.append("catboost")
            model_features_list.append(X_train.columns.to_list())
            logger.info(f"Model {t} appended to MergedModel")

            """
            # Note: calibrated and non_calibrated predictions are not too different.
            # Not integrating into litmus yet
            # Calibrate classifier so predict_proba is aligned with frequentist definition
            cv_params = self.freqai_info["feature_selection"]["cv"]
            base_model = CatBoostClassifier(**ho_best_params)
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_model,
                method="isotonic",
                cv=cv_params["cv_n_splits"],
                n_jobs=cv_params["cv_n_splits"],
                ensemble=True
            )
            logger.info("Fitting calibrated model")
            calibrated_model.fit(X_train, y_train, sample_weight=weight_train.to_numpy().ravel())

            calibrated_proba = calibrated_model.predict_proba(X_test)
            non_calibrated_proba = model.predict_proba(X_test)

            print(np.hstack([non_calibrated_proba, calibrated_proba]))"""

            end_time = time()
            total_time = end_time - start_time
            dk.data["extra_returns_per_train"][f"total_time_{t}"] = total_time

        merged_model = MergedModel(
            model_list=model_list,
            model_type_list=model_type_list,
            model_features_list=model_features_list
        )

        return merged_model
