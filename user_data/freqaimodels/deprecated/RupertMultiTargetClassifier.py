import logging
import sys
from pathlib import Path
from typing import Any, Dict

from catboost import CatBoostClassifier, Pool
from feature_engine.selection import DropCorrelatedFeatures
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class RupertMultiTargetClassifier(BaseClassifierModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        # Swap train and test data
        X_train = data_dictionary["test_features"]
        y_train = data_dictionary["test_labels"]
        weight_train = data_dictionary["test_weights"]
        X_test = data_dictionary["train_features"]
        y_test = data_dictionary["train_labels"]
        weight_test = data_dictionary["train_weights"]

        # Drop features using Greedy Correlated Selection
        if self.freqai_info["feature_selection"]["greedy_selection"].get("enabled", False):
            num_feat = len(X_train.columns)
            logger.info(f"Starting Greedy Feature Selection for {num_feat} features")

            # Get config params
            greedy_selection_params = self.freqai_info["feature_selection"]["greedy_selection"]

            greedy_selection = DropCorrelatedFeatures(
                variables=None, method="pearson",
                threshold=greedy_selection_params["threshold"], missing_values="ignore",
                confirm_variables=False)

            X_train = greedy_selection.fit_transform(X_train, y_train)
            X_test = greedy_selection.transform(X_test)

            num_remaining = len(X_train.columns)
            num_dropped = len(greedy_selection.features_to_drop_)
            logger.info(f"Dropping {num_dropped} correlated features using greedy. "
                        f"{num_remaining} remaining.")

        # Define model
        cbc = CatBoostClassifier(
            train_dir=Path(dk.data_path),
            **self.model_training_parameters
        )

        """X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]"""

        eval_sets = [None] * y_train.shape[1]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            eval_sets = [None] * y_test.shape[1]

            for i in range(y_test.shape[1]):
                eval_sets[i] = Pool(
                    data=X_test,
                    label=y_test.iloc[:, i],
                    weight=weight_test
                )

        init_model = self.get_init_model(dk.pair)

        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y_train.shape[1]

        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append({
                'eval_set': eval_sets[i], 'init_model': init_models[i],
                'log_cout': sys.stdout, 'log_cerr': sys.stderr,
            })

        model = FreqaiMultiOutputClassifier(estimator=cbc)
        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
        if thread_training:
            model.n_jobs = y_train.shape[1]
        model.fit(X=X_train, y=y_train, sample_weight=weight_train, fit_params=fit_params)

        return model
