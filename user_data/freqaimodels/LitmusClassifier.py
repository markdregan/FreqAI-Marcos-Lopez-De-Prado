import logging
import numpy as np
import numpy.typing as npt
import pandas as pd

from catboost import CatBoostClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel
# from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from pandas import DataFrame
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.pipeline import Pipeline
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

        fs_est = LogisticRegression(max_iter=1000, n_jobs=1)
        fs = RFECV(fs_est, min_features_to_select=500, scoring="average_precision",
                   step=0.2, cv=5, verbose=2)
        clf = CatBoostClassifier(iterations=500, loss_function="Logloss", allow_writing_files=False,
                                 early_stopping_rounds=50, verbose=True)
        sampling = SMOTE()

        pipe = Pipeline([
            ("feature_selection", fs),
            ('sampling', sampling),
            ("clf", clf),
        ], verbose=True)

        ovr_estimator = OneVsRestClassifier(pipe, n_jobs=-1)

        ovr_estimator.fit(X_train, y_train)

        # Classification Report
        y_pred = ovr_estimator.predict(X_test)
        print(classification_report(y_test, y_pred))

        # TODO: PR Curve
        # TODO: ROC Curve
        # TODO: Feature Importance
        # TODO: Add IMB Learn

        """cbr = CatBoostClassifier(
            allow_writing_files=False,
            loss_function='MultiClass',
            **self.model_training_parameters,
        )

        cbr.fit(train_data)"""

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
