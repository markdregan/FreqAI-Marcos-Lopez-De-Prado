# mypy: ignore-errors
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd

from catboost import CatBoostClassifier, Pool, EFstrType
from feature_engine.selection import SmartCorrelatedSelection
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.litmus.model_helpers import MergedModel
from freqtrade.litmus.db_helpers import save_df_to_db
from pandas import DataFrame
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from time import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class LitmusMultiTargetClassifier(IFreqaiModel):
    """
    Model that supports multiple classifiers trained separately with
    different features and model weights.
    """

    def train(
            self, unfiltered_dataframe: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_dataframe: Full dataframe for the current training period
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info("-------------------- Starting training " f"{pair} --------------------")

        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_dataframe,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_dataframe["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_dataframe["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to "
                    f"{end_date}--------------------")
        # split data into train/test data.
        data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions", 0) or not self.live:
            dk.fit_labels()
        # normalize all data based on train_dataset only
        data_dictionary = dk.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f'Training model on {len(dk.data_dictionary["train_features"].columns)}' " features"
        )
        logger.info(f'Training model on {len(data_dictionary["train_features"])} data points')

        """# Pass back mask df with index for subsampling (added my markdregan)
        dk.data_dictionary["tbm_mask"] = unfiltered_dataframe["tbm_mask"]"""

        model = self.fit(data_dictionary, dk)

        end_time = time()

        logger.info(f"-------------------- Done training {pair} "
                    f"({end_time - start_time:.2f} secs) --------------------")

        return model

    def predict(
            self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, **kwargs
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

        self.data_cleaning_predict(dk)

        predictions = self.model.predict(dk.data_dictionary["prediction_features"])
        pred_df = DataFrame(predictions, columns=dk.label_list)

        predictions_prob = self.model.predict_proba(dk.data_dictionary["prediction_features"])
        pred_df_prob = DataFrame(predictions_prob, columns=self.model.classes_)

        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        return (pred_df, dk.do_predict)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """

        models = []

        for t in data_dictionary["train_labels"].columns:
            logger.info(f"Start training for target {t}")

            # Swap train and test data
            X_train = data_dictionary["test_features"]
            y_train = data_dictionary["test_labels"][t]
            weight_train = data_dictionary["test_weights"]
            X_test = data_dictionary["train_features"]
            y_test = data_dictionary["train_labels"][t]
            weight_test = data_dictionary["train_weights"]

            # Smart Correlated Selection
            if self.freqai_info['feature_parameters'].get("use_smart_selection", False):
                logger.info("Starting Smart Feature Selection")

                # Get config params
                smart_selection_params = self.freqai_info['feature_parameters'].get(
                    "smart_selection_params", 0)

                fe_estimator = RandomForestClassifier(
                    n_estimators=smart_selection_params["n_estimators"], n_jobs=4)

                smart_selection = SmartCorrelatedSelection(
                    variables=None, method="pearson",
                    threshold=smart_selection_params["threshold"], missing_values="ignore",
                    selection_method="model_performance", estimator=fe_estimator,
                    scoring=smart_selection_params["scoring"], cv=smart_selection_params["cv"],
                    confirm_variables=False)

                X_train = smart_selection.fit_transform(X_train, y_train)
                X_test = smart_selection.transform(X_test)

                num_dropped = len(smart_selection.features_to_drop_)
                dk.data["extra_returns_per_train"][f"num_features_dropped_{t}"] = num_dropped
                logger.info(f"Dropping {num_dropped} correlated features")

            # Create Pool objs for catboost
            train_data = Pool(data=X_train, label=y_train, weight=weight_train)
            eval_data = Pool(data=X_test, label=y_test, weight=weight_test)

            model = CatBoostClassifier(
                allow_writing_files=True,
                train_dir=Path(dk.data_path / "tensorboard"),
                **self.model_training_parameters
            )
            init_model = self.get_init_model(dk.pair)
            model.fit(X=train_data, eval_set=eval_data, init_model=init_model)

            dk.data["extra_returns_per_train"][f"num_trees_{t}"] = model.tree_count_

            # Compute feature importance & save to db
            feature_imp = model.get_feature_importance(
                data=eval_data, type=EFstrType.LossFunctionChange, prettified=True)
            feature_imp.rename(columns={"Feature Id": "feature_id", "Importances": "importance"},
                               inplace=True)
            feature_imp["pair"] = dk.pair
            feature_imp["train_time"] = time.time()
            feature_imp["model"] = t
            feature_imp["rank"] = feature_imp["importance"].rank(method="first", ascending=False)
            feature_imp.set_index(keys=["model", "train_time", "pair", "feature_id"], inplace=True)
            save_df_to_db(df=feature_imp, table_name="feature_importance_history")

            # Model performance metrics
            encoder = LabelBinarizer()
            encoder.fit(y_train)
            y_test_enc = encoder.transform(y_test)
            y_pred_proba = model.predict_proba(X_test)

            # Model performance metrics
            for i, c in enumerate(encoder.classes_):
                # Area under precision recall curve
                pr_auc = average_precision_score(y_test_enc[:, i], y_pred_proba[:, i])
                logger.info(f"{c} - PR AUC score: {pr_auc}")
                # Max F1 Score and Optimum threshold
                precision, recall, thresholds = precision_recall_curve(
                    y_test_enc[:, i], y_pred_proba[:, i])
                numerator = 2 * recall * precision
                denom = recall + precision
                f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom),
                                      where=(denom != 0))
                max_f1 = np.max(f1_scores)
                max_f1_thresh = thresholds[np.argmax(f1_scores)]
                dk.data['extra_returns_per_train'][f"max_f1_{c}"] = max_f1
                dk.data['extra_returns_per_train'][f"max_f1_threshold_{c}"] = max_f1_thresh
                logger.info(f"{c} - Max F1 score: {max_f1}")
                logger.info(f"{c} - Optimum Threshold Max F1 score: {max_f1_thresh}")

            models.append(model)

        model = MergedModel(models)

        return model
