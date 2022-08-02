import logging

from typing import Any, Dict
from catboost import CatBoostRegressor, Pool
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel


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

        train_data = Pool(
            data=data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )

        test_data = Pool(
            data=data_dictionary["test_features"],
            label=data_dictionary["test_labels"],
            weight=data_dictionary["test_weights"],
        )

        model = CatBoostRegressor(
            allow_writing_files=False,
            **self.model_training_parameters
        )

        # Calculate best model performance using all features
        # Train on data closest to present, test on data in the past
        model.fit(X=test_data, eval_set=train_data)
        logger.info(f"Best model performance: {model.get_best_score()}")
        logger.info(f"Best iteration: {model.best_iteration_}")

        return model
