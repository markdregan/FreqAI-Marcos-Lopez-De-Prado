# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd

# from imblearn.combine import SMOTEENN
from catboost import CatBoostClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from pandas import DataFrame
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Dict, Tuple


logger = logging.getLogger(__name__)


class LitmusBaseClassifier(IFreqaiModel):
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

        X_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        # weight_train = data_dictionary["train_weights"]
        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"]

        fs_est = DecisionTreeClassifier()
        fs = RFECV(fs_est, min_features_to_select=500, scoring="average_precision",
                   step=0.35, cv=3, verbose=2, n_jobs=1)
        sampling = SMOTE(sampling_strategy=0.4, n_jobs=1)
        clf = CatBoostClassifier(iterations=300, loss_function="Logloss", allow_writing_files=False,
                                 early_stopping_rounds=50, verbose=False)

        pipe = Pipeline([
            ("feature_selection", fs),
            ("sampling", sampling),
            ("clf", clf),
        ])

        ovr_estimator = OneVsRestClassifier(pipe, n_jobs=5)

        ovr_estimator.fit(X_train, y_train)

        # Model performance metrics
        encoder = LabelBinarizer()
        encoder.fit(y_train)
        y_test_enc = encoder.transform(y_test)
        y_pred_proba = ovr_estimator.predict_proba(X_test)

        for i, c in enumerate(encoder.classes_):
            logger.debug(f"Model performance scores for: {c}")

            # ROC
            roc_auc = roc_auc_score(y_test_enc[:, i], y_pred_proba[:, i])
            dk.data['extra_returns_per_train'][f"roc_auc_{c}"] = roc_auc
            logger.debug(f"ROC AUC score: {roc_auc}")

            # Average Precision
            avg_precision = average_precision_score(y_test_enc[:, i], y_pred_proba[:, i])
            dk.data['extra_returns_per_train'][f"avg_precision_{c}"] = avg_precision
            logger.debug(f"Average precision score: {avg_precision}")

        # TODO: Feature Importance

        return ovr_estimator
