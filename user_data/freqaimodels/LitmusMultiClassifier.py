# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import time

from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm, Pool, EFstrType
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from user_data.litmus.model_helpers import MergedModel
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from typing import Any, Dict, Tuple
from user_data.litmus import db_helpers


logger = logging.getLogger(__name__)


class LitmusMultiClassifier(IFreqaiModel):
    """
    Base class for regression type models (e.g. Catboost, LightGBM, XGboost etc.).
    User *must* inherit from this class and set fit() and predict(). See example scripts
    such as prediction_models/CatboostPredictionModel.py for guidance.
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
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # Create Pool objs for catboost
            train_data = Pool(data=X_train, label=y_train)
            eval_data = Pool(data=X_test, label=y_test)

            model = CatBoostClassifier(
                iterations=1000, loss_function="MultiClass", allow_writing_files=False,
                early_stopping_rounds=30, task_type="CPU", verbose=False)

            # Feature selection & training
            all_feature_names = np.arange(len(X_train.columns))
            summary = model.select_features(
                X=train_data, eval_set=eval_data, num_features_to_select=500,
                features_for_select=all_feature_names, steps=2,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
                shap_calc_type=EShapCalcType.Approximate, train_final_model=True, verbose=False)

            # Feature importance: All features selected and eliminated
            selected = pd.DataFrame(summary["selected_features_names"], columns=["feature_name"])
            selected["selected"] = True
            eliminated = pd.DataFrame(
                summary["eliminated_features_names"], columns=["feature_name"])
            eliminated["selected"] = False
            feature_df = pd.concat([selected, eliminated], ignore_index=True)
            feature_df["pair"] = dk.pair
            feature_df["train_time"] = time.time()

            # Feature Importance: Scores for selected features
            selected_importances = model.get_feature_importance(
                data=eval_data, type=EFstrType.LossFunctionChange, prettified=True)
            selected_importances["rank"] = selected_importances["Importances"].rank(ascending=False)

            # Join
            feature_df = feature_df.merge(
                selected_importances, how="outer", left_on="feature_name", right_on="Feature Id")
            feature_df.drop(columns="Feature Id", inplace=True)
            feature_df.set_index(keys=["train_time", "pair", "feature_name"], inplace=True)
            db_helpers.save_feature_importance(df=feature_df, table_name="feature_importance")

            # Model performance metrics
            encoder = LabelBinarizer()
            encoder.fit(y_train)
            y_test_enc = encoder.transform(y_test)
            y_pred_proba = model.predict_proba(X_test)

            # Model performance metrics
            for i, c in enumerate(encoder.classes_):
                # Area under ROC
                roc_auc = roc_auc_score(y_test_enc[:, i], y_pred_proba[:, i], average=None)
                dk.data['extra_returns_per_train'][f"roc_auc_{c}"] = roc_auc
                logger.info(f"{c} - ROC AUC score: {roc_auc}")
                # Area under precision recall curve
                pr_auc = average_precision_score(y_test_enc[:, i], y_pred_proba[:, i])
                dk.data['extra_returns_per_train'][f"pr_auc_{c}"] = pr_auc
                logger.info(f"{c} - PR AUC score: {pr_auc}")

            models.append(model)

        model = MergedModel(models)

        end_time = time.time() - start_time
        dk.data['extra_returns_per_train']["time_to_train"] = end_time
        logger.info(f"Time taken to select best features & train model: {end_time} seconds")

        return model
