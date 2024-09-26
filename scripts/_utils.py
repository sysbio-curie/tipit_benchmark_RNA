import yaml
from joblib import Parallel
from tqdm.auto import tqdm
import xgboost as xgb


def read_yaml(fname):
    with open(fname) as yaml_file:
        return yaml.safe_load(yaml_file)


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class CustomXGBoostClassifier(xgb.sklearn.XGBClassifier):
    """
    Custom XGBoost classifier to deal with class imbalance.

    """

    def __init__(self, **kwargs):
        super(CustomXGBoostClassifier, self).__init__(**kwargs)

    def fit(self, X, y, feature_weights=None):
        """
        Fit the classifier to the provided training data.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            Training data.

        y: array of shape (n_samples,)
            Target vector.

        feature_weights: array of shape (n_features,)
            Weights assigned to features. The default is None.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super(CustomXGBoostClassifier, self).set_params(
            scale_pos_weight=(y == 0).sum() / (y == 1).sum()
        )
        super(CustomXGBoostClassifier, self).fit(
            X=X, y=y, feature_weights=feature_weights
        )
        return self