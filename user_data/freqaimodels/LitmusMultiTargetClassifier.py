# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd

from catboost import CatBoostClassifier, Pool, EFstrType, EFeaturesSelectionAlgorithm, EShapCalcType
from feature_engine.selection import (SmartCorrelatedSelection, RecursiveFeatureAddition,
                                      DropHighPSIFeatures, DropCorrelatedFeatures)
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.litmus.model_helpers import MergedModel
from freqtrade.litmus.db_helpers import save_df_to_db
from optuna.integration import CatBoostPruningCallback
from pandas import DataFrame
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from time import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class LitmusMultiTargetClassifier(IFreqaiModel):
    """
    Model that supports multiple classifiers trained separately with
    different features and model weights.
    """

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

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f'Training model on {len(dk.data_dictionary["train_features"].columns)}' " features"
        )
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        """# Pass back mask df with index for subsampling (added my markdregan)
        dk.data_dictionary["tbm_mask"] = unfiltered_dataframe["tbm_mask"]"""

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

        models = []

        for t in data_dictionary["train_labels"].columns:
            logger.info(f"Start training for target {t}")

            # Swap train and test data
            X_train = data_dictionary["test_features"]
            y_train = data_dictionary["test_labels"][t]
            weight_train = data_dictionary["test_weights"]
            X_test = data_dictionary["train_features"]
            y_test = data_dictionary["train_labels"][t]
            weight_test = data_dictionary["train_weights"]

            # Drop High PSI Features
            if self.freqai_info["feature_selection"]["psi_elimination"].get("enabled", False):
                num_feat = len(X_train.columns)
                logger.info(f"Starting PSI Feature Elimination for {num_feat} features")

                # Get config params
                psi_elimination_params = self.freqai_info["feature_selection"]["psi_elimination"]

                psi_elimination = DropHighPSIFeatures(
                    split_col=None, split_frac=psi_elimination_params["split_frac"],
                    split_distinct=False, cut_off=None, switch=False,
                    threshold=psi_elimination_params["threshold"], bins=10,
                    strategy="equal_frequency", min_pct_empty_bins=0.0001, missing_values="ignore",
                    variables=None, confirm_variables=False)

                X_train = psi_elimination.fit_transform(X_train, y_train)
                X_test = psi_elimination.transform(X_test)

                num_remaining = len(X_train.columns)
                num_dropped = len(psi_elimination.features_to_drop_)
                dk.data["extra_returns_per_train"][f"num_dropped_psi_{t}"] = num_dropped
                logger.info(f"Dropping {num_dropped} high psi features. {num_remaining} remaining.")

            # Drop features using Greedy Correlated Selection
            if self.freqai_info["feature_selection"]["greedy_selection"].get("enabled", False):
                num_feat = len(X_train.columns)
                logger.info(f"Starting Greedy Feature Selection for {num_feat} features")

                # Get config params
                greedy_selection_params = self.freqai_info["feature_selection"]["greedy_selection"]

                greedy_selection = DropCorrelatedFeatures(
                    variables=None, method="pearson",
                    threshold=greedy_selection_params["threshold"], missing_values="ignore",
                    confirm_variables=False)

                X_train = greedy_selection.fit_transform(X_train, y_train)
                X_test = greedy_selection.transform(X_test)

                num_remaining = len(X_train.columns)
                num_dropped = len(greedy_selection.features_to_drop_)
                dk.data["extra_returns_per_train"][f"num_dropped_greedy_{t}"] = num_dropped
                logger.info(f"Dropping {num_dropped} correlated features using greedy. "
                            f"{num_remaining} remaining.")

            # Drop features using Smart Correlated Selection
            if self.freqai_info["feature_selection"]["smart_selection"].get("enabled", False):
                num_feat = len(X_train.columns)
                logger.info(f"Starting Smart Feature Selection for {num_feat} features")

                # Get config params
                smart_selection_params = self.freqai_info["feature_selection"]["smart_selection"]

                scs_estimator = RandomForestClassifier(
                    n_estimators=smart_selection_params["n_estimators"], n_jobs=4)

                smart_selection = SmartCorrelatedSelection(
                    variables=None, method="pearson",
                    threshold=smart_selection_params["threshold"], missing_values="ignore",
                    selection_method="model_performance", estimator=scs_estimator,
                    scoring=smart_selection_params["scoring"], cv=smart_selection_params["cv"],
                    confirm_variables=False)

                X_train = smart_selection.fit_transform(X_train, y_train)
                X_test = smart_selection.transform(X_test)

                num_remaining = len(X_train.columns)
                num_dropped = len(smart_selection.features_to_drop_)
                dk.data["extra_returns_per_train"][f"num_dropped_corr_{t}"] = num_dropped
                logger.info(f"Dropping {num_dropped} correlated features. "
                            f"{num_remaining} remaining.")

            # Recursive feature addition
            if self.freqai_info["feature_selection"]["recursive_addition"].get(
                    "enabled", False):
                num_feat = len(X_train.columns)
                logger.info(f"Starting RFA Process for {num_feat} features")

                # Get config params
                elimination_params = self.freqai_info["feature_selection"]["recursive_addition"]

                sbs_estimator = RandomForestClassifier(
                    n_estimators=elimination_params["n_estimators"], n_jobs=4)

                feature_elimination = RecursiveFeatureAddition(
                    estimator=sbs_estimator, scoring=elimination_params["scoring"],
                    cv=elimination_params["cv"], threshold=elimination_params["threshold"],
                    variables=None, confirm_variables=False)

                X_train = feature_elimination.fit_transform(X_train, y_train)
                X_test = feature_elimination.transform(X_test)

                num_remaining = len(X_train.columns)
                num_dropped = len(feature_elimination.features_to_drop_)
                dk.data["extra_returns_per_train"][f"num_dropped_model_{t}"] = num_dropped
                logger.info(f"Dropping {num_dropped} features with poor performance. "
                            f"{num_remaining} remaining.")

            # Create Pool objs for catboost
            train_data = Pool(
                data=X_train,
                label=y_train,
                weight=weight_train
            )
            eval_data = Pool(
                data=X_test,
                label=y_test,
                weight=weight_test
            )

            model = CatBoostClassifier(
                **self.model_training_parameters
            )

            # Feature reduction using catboost routine
            if self.freqai_info["feature_selection"]["catboost_selection"].get("enabled", False):
                num_feat = len(X_train.columns)
                logger.info(f"Starting Catboost Feature Selection for {num_feat} features")

                # Get config params for feature selection
                selection_params = self.freqai_info["feature_selection"]["catboost_selection"]

                # Run feature selection
                all_feature_names = np.arange(len(X_train.columns))
                summary = model.select_features(
                    X=train_data, eval_set=eval_data,
                    num_features_to_select=selection_params["num_features_select"],
                    features_for_select=all_feature_names, steps=selection_params["steps"],
                    algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
                    shap_calc_type=EShapCalcType.Approximate, train_final_model=False)

                selected_features_names = summary["selected_features_names"]

                train_data = Pool(
                    data=X_train.loc[:, selected_features_names],
                    label=y_train,
                    weight=weight_train
                )
                eval_data = Pool(
                    data=X_test.loc[:, selected_features_names],
                    label=y_test,
                    weight=weight_test
                )

            # Hyper parameter optimization
            if self.freqai_info["optuna_parameters"].get("enabled", False):
                logger.info("Starting optuna hyper parameter optimization routine")

                def objective(trial: optuna.Trial) -> float:
                    param = {
                        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                        "depth": trial.suggest_int("depth", 1, 12),
                        "boosting_type": trial.suggest_categorical(
                            "boosting_type", ["Ordered", "Plain"]),
                        "bootstrap_type": trial.suggest_categorical(
                            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                        ),
                    }

                    # Add additional static params from config
                    param.update(self.freqai_info["model_training_parameters"])

                    if param["bootstrap_type"] == "Bayesian":
                        param["bagging_temperature"] = trial.suggest_float(
                            "bagging_temperature", 0, 10)
                    elif param["bootstrap_type"] == "Bernoulli":
                        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

                    optuna_model = CatBoostClassifier(**param)

                    pruning_callback = CatBoostPruningCallback(trial, "MultiClassOneVsAll")
                    optuna_model.fit(
                        X=train_data,
                        eval_set=eval_data,
                        verbose=0,
                        callbacks=[pruning_callback])

                    # evoke pruning manually.
                    pruning_callback.check_pruned()

                    best_score = optuna_model.get_best_score()["validation"]["MultiClassOneVsAll"]

                    return best_score

                # Define name and storage of optuna study
                study_name = f"{t}_{dk.pair}"
                storage_name = "sqlite:///LitmusOptunaStudy.sqlite"

                # Load prior optuna trials (if they exist) create new study otherwise
                study = optuna.create_study(
                    storage=storage_name,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                    study_name=study_name,
                    direction="minimize",
                    load_if_exists=True
                )

                # Check latest trial timestamp & re-run if too old
                try:
                    latest_trial = study.trials_dataframe()["datetime_complete"].max()
                    logger.info(f"Latest optuna trial found {latest_trial}")
                    optuna_refresh = self.freqai_info["optuna_parameters"].get(
                        "optuna_refresh", "2:00:00")
                    # if latest delta > 10
                    if latest_trial < pd.Timestamp.now() - pd.Timedelta(optuna_refresh):
                        # Run a new study (delete old study results first)
                        logger.info("Latest optuna study too old. Running new study")
                        optuna.delete_study(study_name, storage_name)
                        study.optimize(objective, n_trials=100, timeout=600)
                except NameError:
                    # No optuna study exists - run for first time
                    logger.info("No optuna study found. Running new optuna study.")
                    study.optimize(objective, n_trials=100, timeout=600)

                # Retrieve the best params & update model params
                best_params = study.best_params
                logger.info(f"Optuna found best params for {t} {dk.pair}: {best_params}")
                model.set_params(**best_params)

            # Train final model
            init_model = self.get_init_model(dk.pair)
            model.fit(X=train_data, eval_set=eval_data, init_model=init_model)

            dk.data["extra_returns_per_train"][f"num_trees_{t}"] = model.tree_count_

            # Compute feature importance & save to db
            feature_imp = model.get_feature_importance(
                data=eval_data, type=EFstrType.LossFunctionChange, prettified=True)
            feature_imp.rename(columns={"Feature Id": "feature_id", "Importances": "importance"},
                               inplace=True)
            feature_imp["pair"] = dk.pair
            feature_imp["train_time"] = time()
            feature_imp["model"] = t
            feature_imp["rank"] = feature_imp["importance"].rank(method="first", ascending=False)
            feature_imp.set_index(keys=["model", "train_time", "pair", "feature_id"], inplace=True)
            save_df_to_db(df=feature_imp, table_name="feature_importance_history")

            end_time = time()
            total_time = end_time - start_time
            dk.data["extra_returns_per_train"]["total_time"] = total_time

            # Model performance metrics
            encoder = LabelBinarizer()
            encoder.fit(y_train)
            y_test_enc = encoder.transform(y_test)
            y_pred_proba = model.predict_proba(X_test)

            def optimum_f1(precision, recall, thresholds, beta):
                numerator = (1 + beta ** 2) * recall * precision
                denom = recall + (beta ** 2 * precision)
                f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom),
                                      where=(denom != 0))
                max_f1 = np.max(f1_scores)
                max_f1_thresh = thresholds[np.argmax(f1_scores)]

                return max_f1, max_f1_thresh

            # Model performance metrics
            for i, c in enumerate(encoder.classes_):
                # Max F1 Score and Optimum threshold
                precision, recall, thresholds = precision_recall_curve(
                    y_test_enc[:, i], y_pred_proba[:, i])

                fbeta_entry = self.freqai_info["trigger_parameters"].get("fbeta_entry", 1)
                fbeta_exit = self.freqai_info["trigger_parameters"].get("fbeta_exit", 1)

                max_fbeta_entry, fbeta_entry_thresh = optimum_f1(
                    precision, recall, thresholds, beta=fbeta_entry)
                max_fbeta_exit, fbeta_exit_thresh = optimum_f1(
                    precision, recall, thresholds, beta=fbeta_exit)

                dk.data["extra_returns_per_train"][f"max_fbeta_entry_{c}_{t}"] = max_fbeta_entry
                dk.data["extra_returns_per_train"][f"max_fbeta_exit_{c}_{t}"] = max_fbeta_exit
                dk.data["extra_returns_per_train"][f"fbeta_entry_thresh_{c}_{t}"] = \
                    fbeta_entry_thresh
                dk.data["extra_returns_per_train"][f"fbeta_exit_thresh_{c}_{t}"] = fbeta_exit_thresh

                logger.info(f"{c} - Max FBeta exit score: {max_fbeta_exit}")
                logger.info(f"{c} - Max FBeta entry score: {max_fbeta_entry}")
                logger.info(f"{c} - Optimum Threshold FBeta exit score: {fbeta_exit_thresh}")
                logger.info(f"{c} - Optimum Threshold FBeta entry score: {fbeta_entry_thresh}")

            models.append(model)

        model = MergedModel(models)

        return model
