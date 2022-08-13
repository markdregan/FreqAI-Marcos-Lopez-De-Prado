import logging
import numpy as np
import numpy.typing as npt
import pandas as pd

# from imblearn.combine import SMOTEENN
from catboost import CatBoostClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel
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


class LitmusClassifier(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict) -> Any:
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
                   step=0.2, cv=3, verbose=2, n_jobs=1)
        sampling = SMOTE(sampling_strategy=0.4, n_jobs=1)
        clf = CatBoostClassifier(iterations=300, loss_function="Logloss", allow_writing_files=False,
                                 early_stopping_rounds=50, verbose=False)

        pipe = Pipeline([
            ("feature_selection", fs),
            ("sampling", sampling),
            ("clf", clf),
        ])

        ovr_estimator = OneVsRestClassifier(pipe, n_jobs=1)

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
            # dk.data['extra_returns_per_train'][f"roc_auc_{c}"] = roc_auc
            logger.debug(f"ROC AUC score: {roc_auc}")

            # Average Precision
            avg_precision = average_precision_score(y_test_enc[:, i], y_pred_proba[:, i])
            # dk.data['extra_returns_per_train'][f"avg_precision_{c}"] = roc_auc
            logger.debug(f"Average precision score: {avg_precision}")

        # TODO: Feature Importance

        return ovr_estimator

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
