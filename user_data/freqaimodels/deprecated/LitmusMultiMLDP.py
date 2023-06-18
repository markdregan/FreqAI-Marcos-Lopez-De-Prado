import logging
import sys
from pathlib import Path
from typing import Any, Dict

from catboost import CatBoostClassifier, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.litmus.base_models.LitmusFreqaiMultiOutputClassifier import \
    LitmusFreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class LitmusMultiMLDP(BaseClassifierModel):
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

        # Define X, y for train and test
        X_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        weight_train = data_dictionary["train_weights"]

        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"]
        weight_test = data_dictionary["test_weights"]

        # Below: Copied from Freqai template

        cbc = CatBoostClassifier(
            allow_writing_files=True,
            train_dir=Path(dk.data_path),
            **self.model_training_parameters,
        )

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

        model = LitmusFreqaiMultiOutputClassifier(estimator=cbc)
        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
        if thread_training:
            model.n_jobs = y_train.shape[1]
        model.fit(X=X_train, y=y_train, sample_weight=weight_train, fit_params=fit_params)

        return model
