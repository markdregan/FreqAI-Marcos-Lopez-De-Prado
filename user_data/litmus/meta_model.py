##
# Brief:    For an arbitrary freqtrade strategy, train a meta-model that acts as a
#           "critic" and judges if the trade predictions from the primary model
#           should be made or passed on.
# Author:   markdregan@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTENC
from joblib import dump, load
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class MetaModel:

    def __init__(self, data: pd.DataFrame, X_features_num: list,
                 X_features_cat: list, y_pred_col: str, y_true_col: str, sample_weight_col: str):
        """Class to train a MetaModel to be used as a critic of a freqtrade
        primary strategy.
            ----------
            data : DataFrame containing all the data needed for the MetaModel
            X_features_num : Numeric features for the MetaModel
            X_features_cat : Categorical features for the MetaModel
            y_pred_col : Predicted outcome/class from the primary model
            y_true_col : Actual outcome/class
            sample_weight_col : col name to be used to weight prediction training
        """

        # Define all class attributes
        self.data = data.copy()
        self.X_features_num = X_features_num
        self.X_features_cat = X_features_cat
        self.X_features = X_features_num + X_features_cat
        self.y_pred_col = y_pred_col
        self.y_true_col = y_true_col
        self.sample_weight_col = sample_weight_col
        self.cols_to_check = self.X_features + [self.y_pred_col] + [self.y_true_col]
        self.feature_importance = pd.DataFrame()

    def check_data(self):
        """Check for NaNs across the features that the ML model will use."""

        # Check if features contain any NaNs / Infinity
        if (self.data[self.cols_to_check].isin([np.nan, np.inf, -np.inf]).any(axis=None)):
            print('DataFrame contains NaNs which need to be addressed before proceeding \n')
            cols_with_issue = self.data[self.cols_to_check].isin(
                [np.nan, np.inf, -np.inf]).sum()
            cols_with_issue = cols_with_issue[cols_with_issue > 0].sort_values(ascending=False)
            print('List of features with NaNs are: \n')
            print(cols_with_issue)
            print('\nTo remove these NaN samples, call clean_data()')
        else:
            print('No NaNs found in the DataFrame. Ready to train meta model.')

        return self

    def clean_data(self):
        """Remove NaN values from the dataset. Only applied to
        features that are passed to the MetaModel"""

        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        num_na_observations = self.data[self.cols_to_check].isna().any().sum()
        print('Removed %s NaN / Infinity observations from DataFrame' % num_na_observations)
        self.data = self.data.dropna(subset=self.cols_to_check)

        return self

    def eval_primary_model(self):
        """Output simple report showing performance of primary model"""

        # Suppress zero division warning as primary model has only one class (TRUE)
        print(
            classification_report(y_true=self.data[self.y_true_col],
                                  y_pred=self.data[self.y_pred_col],
                                  zero_division=0))

        return self

    def fix_class_imbalance(self):
        """Deal with class imbalance in the training dataset (using SMOTENC)"""

        # Mask vector identifying categorical features
        features_cat_mask = np.in1d(self.X_features, self.X_features_cat)

        sm = SMOTENC(categorical_features=features_cat_mask)
        self.X_train_balanced, self.y_train_balanced = sm.fit_resample(
            self.X_train, self.y_train)

        # Sample weight is added to X_train because SMOTENC does not return
        # the index of the dataframe post rebalance. Must drop it to prevent lookahead.
        self.sample_weight = self.X_train_balanced[self.sample_weight_col].abs().values
        self.X_train_balanced = self.X_train_balanced.drop(columns=self.sample_weight_col)

        return self

    def feature_transform_pipeline(self, feature_selection=False):
        """Sequence of transformations applied to data before passing to ML model"""

        self.categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.preprocessor = ColumnTransformer(transformers=[
            ("cat", self.categorical_transformer, self.X_features_cat),
        ], remainder='passthrough')

        simple_imputer = SimpleImputer(strategy='constant', fill_value=999)

        model = RandomForestClassifier(n_jobs=-1,
                                       criterion='entropy',
                                       class_weight='balanced_subsample')

        sfs = SFS(estimator=model,
                  k_features='best',
                  forward=True,
                  floating=False,
                  scoring='roc_auc',
                  cv=self.cv,
                  n_jobs=-1,
                  verbose=2)

        if feature_selection:
            self.clf = Pipeline(
                steps=[("preprocessor", self.preprocessor),
                       ("simple_imputer", simple_imputer),
                       ("feature_selection", sfs)])
        else:
            self.clf = Pipeline(
                steps=[("preprocessor", self.preprocessor),
                       ("simple_imputer", simple_imputer),
                       ("classifier", model)])

        return self

    def train_model(self):
        """Trains ML classifier on training data"""

        self.feature_transform_pipeline()
        self.clf.fit(self.X_train_balanced,
                     self.y_train_balanced,
                     classifier__sample_weight=self.sample_weight)

        self.X_transformed_features = self.preprocessor.get_feature_names_out()

        return self

    def setup_cross_validation(self, cv_n_splits: int, cv_gap: int):
        """Setup the iterator for walk forward cross validation"""

        cv_n_slpits = cv_n_splits
        cv_gap = cv_gap
        max_train_size = np.int(len(self.data) / cv_n_slpits)
        self.cv = TimeSeriesSplit(n_splits=cv_n_slpits,
                                  gap=cv_gap,
                                  max_train_size=max_train_size)

        return self

    def get_most_important_features(self, cv_n_splits: int, cv_gap: int, cv_sample: int = 10):
        """An iterative procedure to select the most important features for the model
        TODO: Add early stopping when PR is merged on master"""

        self.setup_cross_validation(cv_n_splits, cv_gap)
        self.feature_transform_pipeline(feature_selection=True)

        # Sort data by date before running CV
        data_to_cv = self.data.sort_index(level='date', ascending=True)

        # Fit model with sample weights passed
        sample_weight = data_to_cv[::cv_sample][self.sample_weight_col].abs().values
        self.clf.fit(
            X=data_to_cv[::cv_sample][self.X_features],
            y=data_to_cv[::cv_sample][self.y_true_col],
            feature_selection__sample_weight=sample_weight)

        self.X_transformed_features = self.preprocessor.get_feature_names_out()
        self.most_important_features_idx = list(
            self.clf.named_steps['feature_selection'].k_feature_idx_)

        print('The top most important features are:')
        print(self.X_transformed_features[self.most_important_features_idx])

        return self

    def plot_feature_selection(self, figsize=(15, 8)):
        """Plot the model performance as features are added/removed
        during feature selection process"""

        _ = plot_sfs(
            self.clf.named_steps['feature_selection'].get_metric_dict(),
            kind='ci',
            figsize=figsize)

        plt.title('Feature Selection Process & Model Performance')
        plt.grid()
        plt.show()

        return self

    def run_cross_validation(self, cv_n_splits: int, cv_gap: int):
        """Run cross validation."""

        self.setup_cross_validation(cv_n_splits, cv_gap)
        self.precision_recall_stats = []
        self.roc_stats = []
        self.feature_importance = []
        self.data.loc[:, 'clf_y_pred_proba'] = np.nan
        self.data.loc[:, 'clf_y_pred'] = np.nan

        # Sort data by date before running CV
        data_to_cv = self.data.sort_index(level='date', ascending=True)
        for train_idx, test_idx in self.cv.split(
                X=data_to_cv[self.X_features], y=data_to_cv[self.y_true_col]):

            self.X_train = data_to_cv.iloc[train_idx][self.X_features + [self.sample_weight_col]]
            self.X_test = data_to_cv.iloc[test_idx][self.X_features]
            self.y_train = data_to_cv.iloc[train_idx][self.y_true_col]
            self.y_test = data_to_cv.iloc[test_idx][self.y_true_col]

            cv_train_from = self.X_train.index.get_level_values(
                'date').min().strftime('%Y-%m-%d')
            cv_train_to = self.X_train.index.get_level_values(
                'date').max().strftime('%Y-%m-%d')
            cv_test_from = self.X_test.index.get_level_values(
                'date').min().strftime('%Y-%m-%d')
            cv_test_to = self.X_test.index.get_level_values(
                'date').max().strftime('%Y-%m-%d')

            separator = " "
            self.cv_label = separator.join([
                'Train:', cv_train_from, cv_train_to, 'Test:', cv_test_from,
                cv_test_to
            ])
            print(self.cv_label)

            self.fix_class_imbalance()
            self.train_model()
            self.predict()
            self.precision_recall_stats.append(self.get_precision_recall_stats())
            self.roc_stats.append(self.get_roc_stats())
            self.feature_importance.append(self.get_feature_importance(label=self.cv_label))

        return self

    def predict(self):
        """Generate predictions on test dataset. This is used later for
        evaluation."""

        # Generate predictions from the trained model
        self.clf_y_pred_proba = self.clf.predict_proba(self.X_test)[:, 1]
        self.clf_y_pred = self.clf.predict(self.X_test)

        # Add column to original data frame with CV predictions for later analysis
        self.data.loc[self.X_test.index, 'clf_y_pred_proba'] = self.clf_y_pred_proba
        self.data.loc[self.X_test.index, 'clf_y_pred'] = self.clf_y_pred

        return self

    def get_precision_recall_stats(self) -> tuple:
        """Get P/R stats that can be used for plotting later."""

        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.clf_y_pred_proba)

        return precision, recall, thresholds, self.cv_label

    def plot_precision_recall_curve(self):
        """Plot a precision vs recall curve"""

        # Create precision recall curve
        fig, ax = plt.subplots(figsize=(15, 10))
        for stats in self.precision_recall_stats:
            ax.plot(stats[1], stats[0], label=stats[3])

        # Add axis labels to plot
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        # Display plot
        plt.grid()
        plt.legend()
        plt.show()

        return self

    def get_roc_stats(self) -> tuple:
        """Get stats on receiver operating curve for model"""

        fpr, tpr, roc_threshold = roc_curve(self.y_test, self.clf_y_pred_proba)

        return fpr, tpr, roc_threshold, self.cv_label

    def plot_roc_curve(self):
        """Plot a standard receiver operating curve"""

        # Create ROC curve plot
        fig, ax = plt.subplots(figsize=(15, 10))
        for stats in self.roc_stats:
            ax.plot(stats[0], stats[1], label=stats[3])
        ax.plot([0, 1], [0, 1], 'k--')

        # Add axis labels to plot
        ax.set_title('ROC Curve')
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')

        # Display plot
        plt.grid()
        plt.legend()
        plt.show()

        return self

    def get_feature_importance(self, label: str) -> pd.DataFrame:
        """Get and store feature importances during cross validation"""

        temp_df = pd.concat([
            pd.DataFrame(self.X_transformed_features, columns=['feature_name'
                                                               ]),
            pd.DataFrame(
                self.clf.named_steps['classifier'].feature_importances_,
                columns=['feature_importance'])
        ],
                            axis=1)
        temp_df['source'] = label

        return temp_df

    def plot_feature_importance(self):
        """Show what features are most important to the model based on mean
        decrease in impurity. Plot shows results from all folds of cross validation."""

        temp_df = pd.concat(self.feature_importance, axis=0)
        plot_height = len(self.X_features)
        feature_rank = temp_df.groupby(
            'feature_name')['feature_importance'].median().sort_values(
                ascending=False).index
        sns.color_palette()
        g = sns.pointplot(x="feature_importance",
                          y="feature_name",
                          hue="source",
                          data=temp_df,
                          order=feature_rank,
                          join=False,
                          ci=None)
        g.fig.set_figwidth(7)
        g.fig.set_figheight(plot_height)

        return self

    def plot_probability_returns_scatter(self):
        """Generate scatter plot for model probability vs trade returns"""

        plt.subplots(figsize=(12, 12))
        g = sns.scatterplot(x=self.data.loc[:, self.sample_weight_col],
                            y=self.data.loc[:, 'clf_y_pred_proba'],
                            hue=self.data.loc[:, 'exit_reason'],
                            palette='Set2',
                            alpha=.5,
                            x_jitter=True,
                            y_jitter=True)
        g.axvline(x=0, color='r', linestyle='--', lw=2)

        return self

    def probability_returns_table(self) -> pd.DataFrame:
        """Generate a table showing mean returns across model probability"""

        temp_df = self.data.dropna(subset=['clf_y_pred_proba']).copy()
        temp_df.loc[:, 'clf_y_pred_proba_bin'] = np.floor(
            self.data.loc[:, 'clf_y_pred_proba'] * 10)
        results = temp_df.groupby('clf_y_pred_proba_bin')[self.sample_weight_col].agg(
            ['mean', 'median', 'count'])

        return results

    def cv_classification_report(self):
        """Classification report representing all the predictions from
        all the cross validation folds"""

        # Generate classification report
        temp_df = self.data.loc[self.data['clf_y_pred'].dropna().index]
        print(
            classification_report(temp_df.loc[:, self.y_true_col],
                                  temp_df.loc[:, 'clf_y_pred']))

        return self

    def model_threshold_table(self) -> pd.DataFrame:
        """Return a table showing precision, recall, threshold relationships"""

        temp = []
        for result in self.precision_recall_stats:
            temp.append(
                pd.DataFrame(result[0:3],
                             index=['precision', 'recall', 'threshold'
                                    ]).transpose().set_index('threshold'))

        temp_df = pd.concat(temp, axis=1)
        temp_df['precision_avg'] = temp_df['precision'].mean(axis=1)
        temp_df['recall_avg'] = temp_df['recall'].mean(axis=1)

        temp_df

        return temp_df

    def run_train_on_all_data(self):
        """Method to train a model on the complete dataset"""

        self.X_train = self.data[self.X_features + [self.sample_weight_col]]
        self.y_train = self.data[self.y_true_col]

        self.fix_class_imbalance()
        self.train_model()
        self.feature_importance.append(
            self.get_feature_importance(label='full_dataset'))

        return self

    def set_model_threshold(self, model_threshold: float):
        """Set a model threshold value that corresponds to a preferred
        precision value"""

        self.model_threshold = model_threshold

        return self

    def save_model(self, filename: 'str'):
        """Save model and config to disk"""

        model_and_config = {}
        model_and_config['model'] = self.clf
        model_and_config['model_threshold'] = self.model_threshold
        model_and_config[
            'X_features'] = self.X_features_num + self.X_features_cat

        # Serialize to disk
        filepath = Path('user_data', 'meta_model', filename)
        dump(model_and_config, open(filepath, 'wb'))

        return self

    def load_model(self, filename: 'str'):
        """Loads model from disk"""

        filepath = Path('user_data', 'meta_model', filename)
        model_and_config = load(open(filepath, 'rb'))

        return model_and_config
