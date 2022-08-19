import logging
import numpy as np

from typing import Any, Dict
from catboost import CatBoostRegressor, Pool, EFeaturesSelectionAlgorithm
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel


logger = logging.getLogger(__name__)


class LitmusRegressionModel(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        # Swap train and test
        train_data = Pool(
            data=data_dictionary["test_features"],
            label=data_dictionary["test_labels"],
            weight=data_dictionary["test_weights"],
        )

        test_data = Pool(
            data=data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )

        # Select best features
        model = CatBoostRegressor(
            iterations=1000, loss_function="RMSE", allow_writing_files=False,
            early_stopping_rounds=30, task_type="CPU", verbose=False
        )

        features = data_dictionary["test_features"].columns
        model.select_features(
            X=train_data,
            eval_set=test_data,
            features_for_select=np.arange(len(features)),
            num_features_to_select=500,
            steps=2,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
            train_final_model=True,
            verbose=False,
            plot=False
        )

        return model
