# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import time

from catboost import CatBoostClassifier, Pool, EFeaturesSelectionAlgorithm, EShapCalcType, EFstrType
from featurewiz import FeatureWiz
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.litmus.model_helpers import MergedModel
from freqtrade.litmus.db_helpers import save_df_to_db
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class LitmusMultiTargetClassifier(IFreqaiModel):
    """
    Model that supports multiple classifiers trained separately with
    different features and model weights.
    """

    def train(
            self, unfiltered_dataframe: DataFrame, pair: str, dk: FreqaiDataKitchen
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
        if not self.freqai_info.get('fit_live_predictions', 0) or not self.live:
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

        logger.info(f"--------------------done training {pair}--------------------")

        return model

    def predict(
            self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, first: bool = False
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

        self.data_cleaning_predict(dk, filtered_dataframe)

        predictions = self.model.predict(dk.data_dictionary["prediction_features"])
        pred_df = DataFrame(predictions, columns=dk.label_list)

        predictions_prob = self.model.predict_proba(dk.data_dictionary["prediction_features"])
        pred_df_prob = DataFrame(predictions_prob, columns=self.model.classes_)

        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        return (pred_df, dk.do_predict)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        start_time = time.time()
        models = []

        for t in data_dictionary["train_labels"].columns:
            # Swap train and test data
            X_train = data_dictionary["test_features"]
            y_train = data_dictionary["test_labels"][t]
            X_test = data_dictionary["train_features"]
            y_test = data_dictionary["train_labels"][t]

            # Address class imbalance
            if self.freqai_info['feature_parameters'].get('use_smote', False):
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # Create Pool objs for catboost
            train_data = Pool(data=X_train, label=y_train)
            eval_data = Pool(data=X_test, label=y_test)

            model = CatBoostClassifier(
                allow_writing_files=False,
                **self.model_training_parameters
            )

            # Featurewiz feature selection
            if self.freqai_info["feature_parameters"].get("use_featurewiz", False):
                corr_limit = self.freqai_info["feature_parameters"]["featurewiz_params"].get(
                    "corr_limit", 0.9
                )
                feature_engg = self.freqai_info["feature_parameters"]["featurewiz_params"].get(
                    "feature_engg", ""
                )
                features = FeatureWiz(corr_limit=corr_limit, feature_engg=feature_engg,
                                      category_encoders="", dask_xgboost_flag=False,
                                      nrows=None, verbose=1)
                X_train = features.fit_transform(X_train, y_train)
                X_test = features.transform(X_test)
                # print(features.features)

                # Refresh Pool objs for catboost
                train_data = Pool(data=X_train, label=y_train)
                eval_data = Pool(data=X_test, label=y_test)

            # Feature selection logic
            if self.freqai_info['feature_parameters'].get("use_feature_selection_routine", False):
                # Get config params for feature selection
                feature_selection_params = self.freqai_info['feature_parameters'].get(
                    "feature_selection_params", 0)

                # Run feature selection
                all_feature_names = np.arange(len(X_train.columns))
                summary = model.select_features(
                    X=train_data, eval_set=eval_data,
                    num_features_to_select=feature_selection_params["num_features_select"],
                    features_for_select=all_feature_names, steps=feature_selection_params["steps"],
                    algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
                    shap_calc_type=EShapCalcType.Approximate, train_final_model=True, verbose=True)

                # Loss Graph
                print(summary["loss_graph"]["loss_values"])
                print(np.argmin(summary["loss_graph"]["loss_values"]))
                # Selected Features
                selected = pd.DataFrame(summary["selected_features_names"],
                                        columns=["feature_name"])
                selected["selected"] = True
                # Eliminated Features
                eliminated = pd.DataFrame(
                    summary["eliminated_features_names"], columns=["feature_name"])
                eliminated["selected"] = False
                # Save to Database
                feature_df = pd.concat([selected, eliminated], ignore_index=True)
                feature_df["pair"] = dk.pair
                feature_df["train_time"] = time.time()
                feature_df["model"] = t
                feature_df.set_index(
                    keys=["model", "train_time", "pair", "feature_name"], inplace=True)
                save_df_to_db(df=feature_df, table_name="feature_selection_history")

            else:
                model.fit(X=train_data, eval_set=eval_data)

            # Compute feature importance & save
            feature_imp = model.get_feature_importance(
                data=eval_data, type=EFstrType.LossFunctionChange, prettified=True)
            feature_imp.rename(columns={"Feature Id": "feature_id", "Importances": "importance"},
                               inplace=True)
            feature_imp["pair"] = dk.pair
            feature_imp["train_time"] = time.time()
            feature_imp["model"] = t
            feature_imp["rank"] = feature_imp["importance"].rank(method="first", ascending=False)
            feature_imp.set_index(keys=["model", "train_time", "pair", "feature_id"], inplace=True)
            save_df_to_db(df=feature_imp, table_name="feature_importance_history")

            # Model performance metrics
            encoder = LabelBinarizer()
            encoder.fit(y_train)
            y_test_enc = encoder.transform(y_test)
            y_pred_proba = model.predict_proba(X_test)

            # Model performance metrics
            for i, c in enumerate(encoder.classes_):
                # Area under ROC
                roc_auc = roc_auc_score(y_test_enc[:, i], y_pred_proba[:, i], average=None)
                logger.info(f"{c} - ROC AUC score: {roc_auc}")
                # Area under precision recall curve
                pr_auc = average_precision_score(y_test_enc[:, i], y_pred_proba[:, i])
                logger.info(f"{c} - PR AUC score: {pr_auc}")
                # Max F1 Score and Optimum threshold
                precision, recall, thresholds = precision_recall_curve(
                    y_test_enc[:, i], y_pred_proba[:, i])
                numerator = 2 * recall * precision
                denom = recall + precision
                f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom),
                                      where=(denom != 0))
                max_f1 = np.max(f1_scores)
                max_f1_thresh = thresholds[np.argmax(f1_scores)]
                dk.data['extra_returns_per_train'][f"max_f1_{c}"] = max_f1
                dk.data['extra_returns_per_train'][f"max_f1_threshold_{c}"] = max_f1_thresh
                logger.info(f"{c} - Max F1 score: {max_f1}")
                logger.info(f"{c} - Optimum Threshold Max F1 score: {max_f1_thresh}")

            models.append(model)

        model = MergedModel(models)

        end_time = time.time() - start_time
        dk.data['extra_returns_per_train']["time_to_train"] = end_time
        logger.info(f"Time taken to select best features & train model: {end_time} seconds")

        return model
