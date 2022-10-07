from abc import abstractmethod

from sklearn.base import BaseEstimator


class FeatureSelector(BaseEstimator):
    """Base class of feature selector methods.

    This class should not be used directly.
    """
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model with X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y: array-like of shape (n_samples,)

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    @abstractmethod
    def select(self, **select_params):
        """Performs feature selection.

        Returns
        -------
        self : object
            Returns the selected indices.
        """

    def fit_select(self, X, y=None, **params):
        """Fit the model with X and y and perform feature selection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y: array-like of shape (n_samples,)
        params: dictionary
            Parameters to be used for fit and select steps.

        Returns
        -------
        Returns the selected indices.
        """
        fit_params = {
            key: params[key]
            for key in self._get_fit_params()
            if key in params
        }
        select_params = {
            key: params[key]
            for key in self._get_select_params()
            if key in params
        }

        self.fit(X, y, **fit_params)
        return self.select(**select_params)

    def get_scores(self):
        if hasattr(self, 'feature_importances_'):
            return self.feature_importances_
        elif hasattr(self, 'coef_'):
            return self.coef_
        else:
            raise ValueError("No scores found in estimator.")

    def _get_fit_params(self):
        """Returns a list of parameter names used during `fit`.
        """
        return []

    def _get_select_params(self):
        """Returns a list of parameter names used during `select`.
        """
        return []
