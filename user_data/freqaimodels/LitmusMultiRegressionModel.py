import logging

from catboost import CatBoostRegressor
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel
from sklearn.multioutput import MultiOutputRegressor
from typing import Any, Dict


logger = logging.getLogger(__name__)


class LitmusMultiRegressionModel(BaseRegressionModel):
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

        # Flip test and train so we retain most recent data for training.
        X_test = data_dictionary["train_features"]
        y_test = data_dictionary["train_labels"]
        weight_test = data_dictionary["train_weights"]

        X_train = data_dictionary["test_features"]
        y_train = data_dictionary["test_labels"]
        weight_train = data_dictionary["test_weights"]

        estimator = CatBoostRegressor(
            allow_writing_files=False,
            **self.model_training_parameters
        )

        model = MultiOutputRegressor(estimator, n_jobs=-1)

        # Calculate best model performance using all features
        # Train on data closest to present, test on data in the past
        model.fit(X=X_train, y=y_train, sample_weight=weight_train,
                  eval_set=(X_test, y_test, weight_test))
        logger.info(f"Best model performance: {model.get_best_score()}")
        logger.info(f"Best iteration: {model.best_iteration_}")

        return model
