import logging

from typing import Any, Dict
from catboost import CatBoostRegressor, Pool
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

        """model = CatBoostRegressor(
            allow_writing_files=False,
            **self.model_training_parameters
        )

        # Calculate best model performance using all features
        model.fit(X=train_data, eval_set=test_data)
        best_score_all = model.get_best_score()"""

        # Select best features
        # logger.info(f"Starting feature selection procedure...")
        # start_time = time.time()
        model = CatBoostRegressor(
            allow_writing_files=False,
            **self.model_training_parameters
        )
        model.fit(X=train_data, eval_set=test_data)

        """features = data_dictionary["train_features"].columns
        summary = model.select_features(
            X=train_data,
            eval_set=test_data,
            features_for_select=np.arange(len(features)),
            num_features_to_select=200,
            steps=3,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Approximate,
            train_final_model=True,
            logging_level='Silent',
            plot=False
        )
        features_to_keep = summary["selected_features_names"]
        logger.info(f"Reduced features from {len(features)} to {len(features_to_keep)}")
        end_time = time.time() - start_time
        logger.info(f"Time taken to select best features: {end_time} seconds")

        logger.info(f"Best model performance on all features: {best_score_all}")"""
        logger.info(f"Best model performance on selected features: {model.get_best_score()}")

        # Get params from fitted model

        # Train model on complete dataset using best params and best features

        return model
