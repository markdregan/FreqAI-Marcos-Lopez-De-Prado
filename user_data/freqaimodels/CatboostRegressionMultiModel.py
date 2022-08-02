import logging
import numpy as np
import time

from typing import Any, Dict
from catboost import CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.multioutput import MultiOutputRegressor
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel

logger = logging.getLogger(__name__)


class CatboostRegressionMultiModel(BaseRegressionModel):
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

        cbr = CatBoostRegressor(
            allow_writing_files=False,
            **self.model_training_parameters,
        )

        # Superset of features
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])
        sample_weight = data_dictionary["train_weights"]

        # Select best features for first target variable
        logger.info("Starting feature selection procedure...")
        start_time = time.time()
        summary = cbr.select_features(
            X=X,
            y=y,
            eval_set=(data_dictionary["test_features"], data_dictionary["test_labels"]),
            features_for_select=np.arange(len(X.columns)),
            num_features_to_select=200,
            steps=3,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
            shap_calc_type=EShapCalcType.Approximate,
            train_final_model=False,
            logging_level='Silent',
            plot=False
        )
        features_to_keep = summary["selected_features_names"]
        end_time = time.time() - start_time
        logger.info(f"Time taken to select best features: {end_time} seconds")

        # Get model performance on all features
        model = MultiOutputRegressor(estimator=cbr)
        model.fit(X=X, y=y, sample_weight=sample_weight)
        all_train_score = model.score(X, y)
        all_test_score = model.score(*eval_set)

        # Get model performance on selected features
        X = X[features_to_keep]
        model = MultiOutputRegressor(estimator=cbr)
        model.fit(X=X, y=y, sample_weight=sample_weight)
        train_score = model.score(X, y)
        test_score = model.score(*eval_set)

        logger.info(f"All features: Train score {all_train_score}, Test score {all_test_score}")
        logger.info(f"Selected features: Train score {train_score}, Test score {test_score}")

        return model
