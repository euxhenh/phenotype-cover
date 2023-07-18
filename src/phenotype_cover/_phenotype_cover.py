from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted

from ._base import FeatureSelector
from ._gci_wrapper import GCIPython, GCIWrapper
from ._logger import logger
from ._operations import pairwise_differences


class SetCoverSelectorBase(FeatureSelector):
    """
    Constructs a base class for set cover selector algorithms.
    """

    def __init__(
            self,
            *,
            ordered=True,
            verbose=True,
            operation=lambda x: x.mean(axis=0)):
        self.ordered = ordered
        self.verbose = verbose
        self.operation = operation

    def fit(self, X, y):
        """
        Given a matrix X and a class label vector y, construct a matrix of
        class pairwise differences for every gene and use it to initialize
        a GreedyCoverInstance via a GCIWrapper.

        Parameters
        __________
        X: ndarray, data matrix, shape (n_samples, n_features)
        y: ndarray, class label vector, shape (n_samples,)
        multiplier: int or None, if not None will be used to multiply
            the pairwise differences to allow for finer resolution
            of the element multiplicities in the greedy cover algorithm.
        Mpath: None or str
            Path to a precomputed pairwise matrix.
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            ensure_min_samples=2,
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )

        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ <= 1:
            raise ValueError("At least 2 class labels must be given.")

        self.n_class_pairs_ = self.n_classes_ * (self.n_classes_ - 1)
        if not self.ordered:
            self.n_class_pairs_ //= 2

        # M_ has shape (n_class_pairs, n_features_)
        self.M_, self.index_to_pair_ = pairwise_differences(
            X, y, classes=self.classes_, ordered=self.ordered,
            operation=self.operation)

        return self

    @abstractmethod
    def select(self):
        raise NotImplementedError

    def _more_tags(self):
        return {
            "allow_nan": False,
            "requires_y": True,
        }

    def _get_fit_params(self):
        return ['multiplier']

    def _get_select_params(self):
        return ['coverage', 'max_iters']


class GreedyPC(SetCoverSelectorBase):
    """Constructs a feature selector based on the greedy cover algorithm.

    Given a data matrix X and a vector of class labels y, select
    a subset of features in X, such that the classes are as `far` from
    each other as possible w.r.t these features.

    E.g.
    ```
    >>> gcs = GreedyPC(ordered=True)
    >>> gcs.fit(X, y, multiplier=10)
    >>> transformed_X = gcs.transform(X, coverage=10)
    ```

    Parameters
    __________
    ordered: bool, if True will construct the a pairwise matrix
        of ordered pairs
    verbose: bool, whether to show info.
    use_python: bool, if True will switch to a Python implementation
        of the greedy cover algorithm. This is here just for
        debugging purposes as the C++ implementation is much faster.
    operation: callable, operation to use when constructing the class vector.

    Attributes
    __________
    n_samples_: int
        Number of samples seen during ``fit``
    n_features_: int
        Number of features seen during ``fit``
    classes_: ndarray of shape (n_classes,)
        Classes seen in the class label vector during ``fit``
    n_classes_: int
        Number of classes seen in the class label vector during ``fit``
    n_class_pairs_: int
        Number of pairs of classes that get constructed. This
        equals nchoosek(n_classes_, 2) if ordered is False,
        else n_classes_ * (n_classes - 1)
    M_: ndarray of shape (n_class_pairs_, n_features_)
        The pairwise difference matrix
    index, index_to_pair_: dict
        Dictionary that maps row indices in M_ to a pair of
        (ordered) classes
    multiplier: int
        Multiplier used to get a finer resolution when constructing
        multiset multiplicities
    coverage: int or ndarray of shape (n_class_pairs_)
        Coverage requested during ``predict``
    max_iters: int
        Maximum number of features to select during ``predict``
    n_outputs_: int
        Number of features selected during ``predict``
    n_pairs_with_incomplete_cover_: int
        Number of pairs (elements) that could not be covered
        to the desired coverage during ``predict``
    """

    def __init__(
            self,
            *,
            ordered=True,
            verbose=True,
            use_python=False,
            multiplier=None,
            operation=lambda x: x.mean(axis=0)):
        super().__init__(ordered=ordered, verbose=verbose, operation=operation)
        self.use_python = use_python
        self._multiplier = multiplier

    def select(self, coverage, *, max_iters=0):
        """
        Returns the indices of the selected features given a certain coverage.

        Parameters
        __________
        coverage: int or list, in case of a list will apply a specific
            coverage to each element.
        max_iters: maximum number of iterations (features) to return.
            A value of 0 or None means no limit.
        """
        check_is_fitted(self)
        self.multiplier = self._multiplier

        self.coverage = coverage
        self.max_iters = max_iters
        # max_iters == 0 means no limit on the number of iterations
        if max_iters is None or max_iters < 0:
            max_iters = 0

        solution = self._gci_wrapper.predict(coverage, max_iters)
        self.solution = solution
        self.feature_importances_ = np.zeros(self.n_features_)
        self.feature_importances_[solution] = np.arange(
            len(solution) + 1, 1, -1)
        if self.n_pairs_with_incomplete_cover_ > 0 and self.verbose:
            logger.warning("Could not cover "
                           f"{self.n_pairs_with_incomplete_cover_} elements.")

        self.n_outputs_ = len(solution)
        logger.info(f"Selected {self.n_outputs_} multisets.")

        return self.solution

    def plot_progress(self):
        """
        Plots the number of remaining elements to be covered,
        and the coverage reached for every feature selected.
        """
        fig, ax1 = plt.subplots()
        ax1.plot(self._gci_wrapper.n_elements_remaining_)
        ax1.set_ylabel('N remaining elements')
        ax1.set_xlabel('N features')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        coverage_until = self._gci_wrapper.coverage_until_
        ax2.plot(coverage_until, color='red')
        ax2.set_ylabel('Coverage')
        ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()
        return ax1

    @property
    def n_elements_remaining_per_iter_(self):
        return self._gci_wrapper.n_elements_remaining_

    @property
    def coverage_per_iter_(self):
        return self._gci_wrapper.coverage_until_

    def feature_coverage(self, index):
        """
        Returns a heatmap of pair coverages for the feature given at index.

        Parameters
        __________
        index: int, feature index in X
        effective: bool, if True will plot the corrected coverage
            of the feature, i.e., after stronger features may have been
            selected
        """
        check_is_fitted(self)

        elements, multiplicity = self._gci_wrapper[index]
        heatmap = np.zeros((self.n_classes_, self.n_classes_))

        for element, mult in zip(elements, multiplicity):
            i1, i2 = self.index_to_pair_[element]
            heatmap[i1][i2] = mult

        return heatmap

    def max_coverage(self):
        """
        Returns a heatmap of max pair coverages.
        """
        check_is_fitted(self)
        heatmap = np.zeros((self.n_classes_, self.n_classes_))
        max_coverage = self._gci_wrapper.max_coverage_

        for key in self.index_to_pair_:
            p1, p2 = self.index_to_pair_[key]
            heatmap[p1][p2] = max_coverage[key]

        return heatmap

    @property
    def n_pairs_with_incomplete_cover_(self):
        check_is_fitted(self)
        return len(self._gci_wrapper.elements_incomplete_cover_)

    @property
    def pairs_with_incomplete_cover_(self):
        """
        Returns pairs of classes that were not covered fully by the
        algorithm. Uses the original class labels.
        """
        check_is_fitted(self)
        int_pairs = [self.index_to_pair_[m]
                     for m in self._gci_wrapper.elements_incomplete_cover_]
        return [(self.classes_[i], self.classes_[j]) for i, j in int_pairs]

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, m):
        """
        Triggers a change on GCIWrapper everytime multiplier is changed
        by re-calling ``fit`` and re-constructing the multisets from M.
        """
        if m is not None and m <= 0:
            raise ValueError("Multiplier must be positive.")

        self._multiplier = m
        if self.use_python:
            logger.info("Using Python")
            self._gci_wrapper = GCIPython(self.verbose)
        else:
            self._gci_wrapper = GCIWrapper(
                self.M_.shape[0], self._multiplier, self.verbose)
        self._gci_wrapper.fit(self.M_)


class CEMPC(SetCoverSelectorBase):
    """Set cover based on the cross entropy method.

    Parameters
    __________
    ordered: bool, if True will construct the a pairwise matrix
        of ordered pairs
    verbose: bool, whether to show info.
    max_iters: int
        Maximum number of iterations for the cross entropy method.
    N: int
        Number of random samples to draw per iteration.
    rho: float in (0, 1)
        quantile to use when selecting gamma_hat.
    alpha: float:
        Parameter that controls the tradeoff between coverage desired
        and the number of features selected. If None, will use
        alpha = coverage * 10 / n_features_
    smoothing_parameter: float
        Parameter to use during exponential smoothing update. If None,
        will not apply smoothing.
    eps: float
        Tolerance to use for stopping the algorithm.
    patience: int
        Number of iterations to wait before stopping the algorithm.
    operation: callable, operation to use when constructing the class vector.
    """

    def __init__(
            self,
            *,
            ordered=True,
            verbose=True,
            max_iters=500,
            rs=1000,
            rho=0.1,
            alpha=None,
            smoothing_parameter=0.7,
            eps=1e-4,
            patience=5,
            operation=lambda x: x.mean(axis=0)):
        super().__init__(ordered=ordered, verbose=verbose, operation=operation)
        self.max_iters = max_iters
        self.rs = rs
        self.rho = rho
        self.alpha = alpha
        self.smoothing_parameter = smoothing_parameter
        self.eps = eps
        self.patience = patience

    def select(self, coverage):
        check_is_fitted(self)
        # round
        # self.M_ = np.round(self.M_).astype(int)
        self.pairs_with_incomplete_cover_ = np.argwhere(
            self.M_.sum(axis=1) < coverage).flatten()
        self.n_pairs_with_incomplete_cover_ = self.pairs_with_incomplete_cover_.size
        if self.n_pairs_with_incomplete_cover_ > 0:
            logger.info(
                f"Cannot cover {self.n_pairs_with_incomplete_cover_} elements.")
        self.coverage = coverage
        self.feature_importances_ = np.zeros(self.n_features_)

        # initial probabilities
        v_hat = np.full_like(self.feature_importances_, 1/2)
        no_change_iter = 0

        self.coverages_ = []
        self.average_n_features_selected_ = []
        ordered_features_ = []

        for iter in range(self.max_iters):
            # draw random samples
            samples = np.random.binomial(1, v_hat, (self.rs, self.n_features_)).astype(float)
            # compute scores for each sample
            scores = self._score(samples, self.alpha)
            # find quantile
            gamma_hat = np.quantile(scores, 1 - self.rho, interpolation='higher')
            prev_vhat = v_hat.copy()
            # update v-hat
            v_hat = ((scores >= gamma_hat)[:, np.newaxis] * samples).sum(axis=0)
            v_hat /= (scores >= gamma_hat).sum()
            # smoothed update
            if self.smoothing_parameter is not None:
                v_hat = (
                    self.smoothing_parameter * v_hat +
                    (1 - self.smoothing_parameter) * prev_vhat
                )
            # check if converged
            # logger.info(v_hat.max())
            # logger.info(np.abs(v_hat - prev_vhat).max())
            if np.abs(v_hat - prev_vhat).max() <= self.eps:
                no_change_iter += 1
                if no_change_iter == self.patience:
                    logger.info(f"Converged with eps={self.eps}.")
                    break
            else:
                no_change_iter = 0

            new_added = (
                set(np.argwhere(v_hat > 0.98).flatten().tolist()) -
                set(ordered_features_)
            )
            if len(new_added) > 0:
                ordered_features_ += list(new_added)
            high_prob_feats = np.argwhere(v_hat > 0.98).flatten()
            # logger.info(f"{high_prob_feats.size} high probability features.")
            # logger.info(f"{np.argwhere(v_hat > 0.5).flatten()}")
            per_element_coverage = self.M_[:, high_prob_feats].sum(axis=1)
            # logger.info(f"{per_element_coverage.min()} smallest coverage.")

        high_prob_feats = np.argwhere(v_hat > 0.98).flatten()
        assert set(high_prob_feats).issubset(set(ordered_features_))
        ordered_features_ = [i for i in ordered_features_
                                  if i in high_prob_feats]

        self.feature_importances_[ordered_features_] = np.arange(
            len(ordered_features_) + 1, 1, -1)

        logger.info(f"{high_prob_feats.size} high probability features.")
        per_element_coverage = self.M_[:, high_prob_feats].sum(axis=1)
        self.min_coverage = per_element_coverage.min()
        logger.info(f"{self.min_coverage} smallest coverage.")
        logger.info(f"{per_element_coverage.mean()} average coverage.")
        self.coverages_ = np.asarray(self.coverages_)
        self.solution = np.asarray(ordered_features_)
        self.average_n_features_selected_ = np.asarray(self.average_n_features_selected_)

        return self

    def _score(self, samples, alpha=None):
        """Score function for given random samples.

        samples: array-like of shape (N, n_features)
        alpha: hyperparameter determining trade-off between coverage
            and number of genes picked.

        Returns:
        scores: np.ndarray of shape (N,)
        """
        element_by_sample = self.M_ @ samples.T
        # Clip to desired coverage
        element_by_sample = np.clip(element_by_sample, 0, self.coverage)
        coverage_per_sample = element_by_sample.min(axis=0)
        genes_picked_per_sample = samples.sum(axis=1)
        if alpha is None:
            alpha = self.coverage / self.n_features_
        self.coverages_.append(coverage_per_sample.mean())
        self.average_n_features_selected_.append(genes_picked_per_sample.mean())
        return coverage_per_sample - alpha * genes_picked_per_sample
