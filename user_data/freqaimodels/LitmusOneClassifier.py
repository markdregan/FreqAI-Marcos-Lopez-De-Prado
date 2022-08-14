# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import time

# from imblearn.combine import SMOTEENN
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm, Pool, EFstrType
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from user_data.litmus import model_helpers
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, precision_recall_curve, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from typing import Any, Dict, Tuple


logger = logging.getLogger(__name__)


class LitmusOneClassifier(IFreqaiModel):
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

        # Swap train and test data
        X_train = data_dictionary["test_features"]
        y_train = data_dictionary["test_labels"]
        X_test = data_dictionary["train_features"]
        y_test = data_dictionary["train_labels"]

        # Address class imbalance
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Create Pool objs for catboost
        train_data = Pool(data=X_train_res, label=y_train_res)
        eval_data = Pool(data=X_test, label=y_test)

        clf = CatBoostClassifier(
            iterations=1000, loss_function="MultiClass", allow_writing_files=False,
            early_stopping_rounds=20, task_type="CPU", verbose=False)

        # Feature selection & training
        start_time = time.time()
        all_feature_names = np.arange(len(X_train.columns))
        clf.select_features(
            X=train_data, eval_set=eval_data, num_features_to_select=500,
            features_for_select=all_feature_names, steps=3,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange,
            shap_calc_type=EShapCalcType.Approximate, train_final_model=True, verbose=False)
        end_time = time.time() - start_time
        dk.data['extra_returns_per_train']["time_to_train"] = end_time
        logger.info(f"Time taken to select best features & train model: {end_time} seconds")

        # Model performance metrics
        encoder = LabelBinarizer()
        encoder.fit(data_dictionary["train_labels"])
        y_test_enc = encoder.transform(data_dictionary["test_labels"])
        y_pred_proba = clf.predict_proba(data_dictionary["test_features"])

        # Model performance metrics
        for i, c in enumerate(encoder.classes_):
            # AUC per class
            roc_auc = roc_auc_score(y_test_enc[:, i], y_pred_proba[:, i])
            dk.data['extra_returns_per_train'][f"roc_auc_{c}"] = roc_auc
            logger.debug(f"{c} - ROC AUC score: {roc_auc}")

            # Threshold values to achieve min precision value
            precision, recall, threshold = precision_recall_curve(
                y_test_enc[:, i], y_pred_proba[:, i])

            for p in [0.1, 0.3, 0.5]:
                t = model_helpers.get_threshold(precision, threshold, p)
                dk.data['extra_returns_per_train'][f"threshold_for_precision_{p}_{c}"] = t
                logger.debug(f"{c} - Threshold for precision {p}: {t}")

                r = recall_score(y_test_enc[:, i], y_pred_proba[:, i] >= t)
                logger.debug(f"{c} - Recall for precision {p}: {r}")

                f1 = f1_score(y_test_enc[:, i], y_pred_proba[:, i] >= t)
                logger.debug(f"{c} - F1 score for precision {p}: {f1}")

        # Feature Importance
        feat_imp_df = clf.get_feature_importance(
            data=eval_data, type=EFstrType.LossFunctionChange, prettified=True,
            shap_calc_type="Approximate")
        feat_imp_df["pair"] = dk.pair
        feat_imp_df["train_time"] = time.time()
        feat_imp_df.set_index(keys=["pair", "train_time"], inplace=True)
        # TODO: Save to db

        return clf
