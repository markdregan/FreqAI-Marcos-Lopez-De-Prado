import logging

from typing import Any, Dict
from catboost import CatBoostRegressor, Pool
from freqtrade.freqai.prediction_models.BaseRegressionModel import BaseRegressionModel

logger = logging.getLogger(__name__)


class CatboostRegressionModel(BaseRegressionModel):
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
            **self.model_training_parameters,
        )

        """# Select best features for first target variable
        logger.info(f"Starting feature selection procedure...")
        start_time = time.time()
        features = data_dictionary["train_features"].columns
        summary = model.select_features(
            X=train_data,
            eval_set=test_data,
            features_for_select=np.arange(len(features)),
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
        logger.info(f"Time taken to select best features: {end_time} seconds")"""

        # Get model performance on full dataset
        model.fit(X=train_data, eval_set=test_data)
        logger.info(f"All features: Best model scores: {model.get_best_score()}")

        """# Recreate pool objects
        train_data = Pool(
            data=data_dictionary["train_features"].loc[:, features_to_keep],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )

        test_data = Pool(
            data=data_dictionary["test_features"].loc[:, features_to_keep],
            label=data_dictionary["test_labels"],
            weight=data_dictionary["test_weights"],
        )

        # Get model performance on smaller selected features dataset
        model.fit(X=train_data, eval_set=test_data)
        logger.info(f"Selected features: Best model scores: {model.get_best_score()}")
        """

        return model
