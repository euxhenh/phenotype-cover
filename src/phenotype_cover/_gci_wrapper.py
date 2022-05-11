import numpy as np
from multiset_multicover import GreedyCoverInstance
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._logger import logger


class GCIWrapper(BaseEstimator):
    def __init__(self, n_elements, multiplier=None, verbose=True):
        """
        Constructs a wrapper for a GreedyCoverInstance that includes
        different functionality such as plotting.

        Args:
            n_elements: int, number of elements in the greey cover
            multiplier: numeric or None. If not None, will multiply each
                element of M with this value, effectively increasing the
                multiplicity of each value. This can be applied when finer
                resolution over the multiplities is needed in case when
                they are represented as float values. Note, the coverage
                factor requested will also be multiplied by this same factor.
            verbose: bool, whether to show info.

        Attributes:
            n_elements: int, total number of elements, i.e., rows of M
            n_multisets_: int, number of multisets actually used, i.e.
                columns with a single nonzero value in M
            max_coverage_: ndarray, maximum possible coverage for every
                element
            multisets_incomplete_cover_: ndarray, indices of multisets
                that could not be covered to the desired coverage
            coverage_until_: ndarray, coverage obtained after every
                iteration. Has shape (solution.size,).
            n_elements_remaining_: ndarray, number of elements remaining
                after every iteration. Has shape (solution.size,)

        """
        if multiplier is not None and multiplier < 0:
            raise ValueError("Cannot have a negative multiplier.")
        self.n_elements = n_elements
        self._gci = GreedyCoverInstance(n_elements)
        self.multiplier = multiplier
        self.verbose = verbose

    def __getitem__(self, i):
        """
        Returns a pair of arrays corresponding to the elements and
        multiplicities of feature i.
        """
        # First convert to the inner index
        i = self._real_index_to_inner(i)
        if i is None:
            return np.array([], dtype=int), np.array([])
        elements, multiplicity = self._gci.__getitem__(i)
        elements = np.array(elements)
        multiplicity = self._divide_for_multiplier(multiplicity)
        return elements, multiplicity

    def effective_at(self, i):
        """
        Returns a pair of arrays corresponding to the elements and
        effective multiplicities of feature i. The effective multiplicity
        of a feature is the actual multiplicities that were needed
        from that feature at the moment of selection. If the feature was
        not selected, then this will return the multiplicities that remained
        for that feature at the end of the cover.
        """
        i = self._real_index_to_inner(i)
        if i is None:
            return np.array([], dtype=int), np.array([])
        elements, multiplicity = self._gci.effective_at(i)
        elements = np.array(elements)
        multiplicity = self._divide_for_multiplier(multiplicity)
        return elements, multiplicity

    def fit(self, M):
        """
        Given a matrix M, construct a greedy cover instance by considering
        the columns of M as multisets. Will skip all-zero columns and
        save the "true" indices of the columns under ``self._true_indices``.

        Args:
            M: np.ndarray, the data matrix. Has shape (n_elements, n_multisets)
        """
        M = self._multiply_for_multiplier(M)
        M = np.round(M).astype(int)

        # Determine features with non-zero entries and push them as multisets.
        self._true_indices = []
        for i, col in enumerate(M.T):
            elements = col.nonzero()[0]
            if elements.size > 0:
                multiplicity = col[elements]
                self._true_indices.append(i)
                self._gci.add_multiset(
                    elements.tolist(), multiplicity.tolist())

        self._true_indices = np.array(self._true_indices)
        self.n_multisets_ = len(self._true_indices)
        if self.verbose:
            logger.info(f"Saved {self.n_multisets_}/{M.shape[1]} multisets.")

    def predict(self, coverage, max_iters=0):
        """
        Returns the indices of the selected features given
        a certain coverage.

        Args:
            coverage: int or list, in case of a list will apply a specific
                coverage to each element.
            max_iters: maximum number of iterations (features) to return.
                A value of 0 or None means no limit.
        """
        check_is_fitted(self)
        coverage = self._multiply_for_multiplier(coverage)
        # Need to convert to Pythonic list in this case, since
        # GreedyCoverInstance instance only accepts python lists.
        if isinstance(coverage, np.ndarray):
            coverage = coverage.tolist()
        # Do not forget to convert back to the true indices
        return self._true_indices[self._gci.cover(coverage, max_iters)]

    def fit_predict(self, M, coverage, max_iters=0):
        """
        Given a matrix M, construct a greedy cover instance by considering
        the columns of M as multisets. Will skip all-zero columns and
        save the "true" indices of the columns under ``self._true_indices``.
        Then returns the indices of the selected features given
        a certain coverage.

        Args:
            M: np.ndarray, the data matrix. Has shape (n_elements, n_multisets)
            coverage: int or list, in case of a list will apply a specific
                coverage to each element.
            max_iters: maximum number of iterations (features) to return.
                A value of 0 or None means no limit.
        """
        self.fit(M)
        return self.predict(coverage, max_iters)

    @property
    def max_coverage_(self):
        """
        The total multiplicity of each element across all multisets.
        """
        return self._divide_for_multiplier(self._gci.max_coverage_)

    @property
    def elements_incomplete_cover_(self):
        """
        Array of indices corresponding to elements that could not
        be covered to the desired coverage.
        """
        return np.array(self._gci.multisets_incomplete_cover_)

    @property
    def coverage_until_(self):
        """
        Array of coverage factors up until and including every
        multiset selected. Has shape (n_solution,)
        """
        return self._divide_for_multiplier(self._gci.coverage_until_)

    @property
    def n_elements_remaining_(self):
        """
        Array of integers corresponding to the number of elements
        that remain up until a multiset has been selected.
        Has shape (n_solution,).
        """
        return np.array(self._gci.n_elements_remaining_)

    def _real_index_to_inner(self, i):
        """
        Converts a real index to an inner index. If the real index was
        not used to construct GCI, returns None.
        """
        check_is_fitted(self)
        if i in self._true_indices:
            return (self._true_indices == i).nonzero()[0][0]
        return None

    def _divide_for_multiplier(self, array):
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = array.astype(float).copy()
        if self.multiplier is not None:
            array /= self.multiplier
        return array

    def _multiply_for_multiplier(self, array):
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = array.copy()
        if self.multiplier is not None:
            array *= self.multiplier
        return array


class GCIPython(BaseEstimator):
    def __init__(self, verbose=True, **kwargs):
        """
        Constructs a wrapper for a GreedyCoverInstance based on a Python
        implementation. Results are identical to GCIWrapper above.
        Use mainly for debugging purposes.

        Args:
            verbose: bool, whether to show info.

        Attributes:
            n_elements: int, total number of elements, i.e., rows of M
            n_multisets_: int, number of multisets actually used, i.e.
                columns with a single nonzero value in M
            max_coverage_: ndarray, maximum possible coverage for every
                element
            multisets_incomplete_cover_: ndarray, indices of multisets
                that could not be covered to the desired coverage
            coverage_until_: ndarray, coverage obtained after every
                iteration. Has shape (solution.size,).
            n_elements_remaining_: ndarray, number of elements remaining
                after every iteration. Has shape (solution.size,)

        """
        self.verbose = verbose

    def __getitem__(self, i):
        check_is_fitted(self)
        if i in self._nonzero_cols:
            return self._M_trimmed[np.where(self._nonzero_cols == i)[0][0]]
        return np.zeros(self.n_elements_.size, dtype=self._M_trimmed.dtype)

    def effective_at(self, i):
        check_is_fitted(self)
        if not hasattr(self, 'effective'):
            raise ValueError("No coverage has been specified.")

        i = self._real_index_to_inner(i)
        if i is None or i not in self._solution:
            return np.zeros(
                self.effective_[0].size, dtype=self.effective_.dtype)

        index = np.where(self._solution == i)[0][0]
        return self.effective_[index]

    def fit(self, M, multiplier=None):
        """
        M: array, rows are elements, columns will be used to construct
            multisets.
        multiplier: numeric or None. If not None, will multiply each
            element of M with this value, effectively increasing the
            multiplicity of each value. This can be applied when finer
            resolution over the multiplities is needed in case when
            they are represented as float values. Note, the coverage
            factor requested will also be multiplied by this same factor.
        """
        if multiplier is not None and multiplier < 0:
            raise ValueError("Cannot have a negative multiplier.")
        self.multiplier_ = multiplier
        self.n_elements = M.shape[0]

        M = self._correct_x_multiplier(M)
        M = np.round(M)

        if M.min() < 0:
            raise ValueError("M cannot contain negative values.")

        self._nonzero_cols = np.where(~np.all(M == 0, axis=0))[0]
        self._M_trimmed = M[:, self._nonzero_cols]
        self.n_multisets_ = len(self._nonzero_cols)
        if self.verbose:
            logger.info(f"Saved {self.n_multisets_}/{M.shape[1]} multisets.")

    @property
    def max_coverage_(self):
        """
        Maximum number of times an element can be covered.
        """
        check_is_fitted(self)
        return self._M_trimmed.sum(axis=1)

    def predict(self, coverage, max_iters=0):
        """
        Returns the indices of the selected features given
        a certain coverage.

        Args:
            coverage: int or list, in case of a list will apply a specific
                coverage to each element.
            max_iters: maximum number of iterations (features) to return.
                A value of 0 or None means no limit.
        """
        self.coverage = coverage
        check_is_fitted(self)
        coverage = self._correct_x_multiplier(coverage)
        self.effective_coverage_ = coverage
        self.max_iters = max_iters

        def stop(x, iter, to_cover):
            if max_iters > 0:
                if iter >= max_iters:
                    logger.info("Reached maximum number of iterations.")
                    return True
            if x.max() == 0:
                logger.info("No more nonzero multisets remain.")
                return True
            if to_cover.max() == 0:
                logger.info("Finished cover.")
                return True
            return False

        temp_M = np.ascontiguousarray(self._M_trimmed.copy())
        temp_M = np.clip(temp_M, 0, coverage)

        if isinstance(coverage, int):
            to_cover = np.full(temp_M.shape[0], coverage, dtype=temp_M.dtype)
        else:
            to_cover = coverage.copy()
        to_cover = np.minimum(to_cover, self.max_coverage_)

        self._solution = []
        self.effective_ = []
        self.coverage_until_ = []
        self.n_elements_remaining_ = []
        iter = 0
        __total_col_sum = self._M_trimmed.sum(axis=0)

        while not stop(temp_M, iter, to_cover):
            # Find multiset with highest value.
            # In case of a collision, take the multiset with the highest
            # total value as determined by _M_trimmed.
            __col_sum = temp_M.sum(axis=0)
            __top_val = np.max(__col_sum)
            __achieve_top_val = np.argwhere(__col_sum == __top_val).flatten()
            if __achieve_top_val.size == 1:
                next_mset = __achieve_top_val.item()
            else:
                __best_total_val = np.argmax(__total_col_sum[__achieve_top_val])
                next_mset = __achieve_top_val[__best_total_val]

            cur_mset_vals = temp_M[:, next_mset].copy()
            to_cover = np.maximum(0, to_cover - cur_mset_vals)
            if isinstance(coverage, np.ndarray):
                self.coverage_until_.append(np.min(coverage - to_cover))
            else:
                self.coverage_until_.append(coverage - to_cover.max())
            self.n_elements_remaining_.append(np.sum(to_cover != 0))

            self.effective_.append(cur_mset_vals)
            self._solution.append(next_mset)
            # Correct the multiplicity matrix by subtracting
            # the selected multiset
            temp_M = np.minimum(temp_M, to_cover.reshape(-1, 1))
            # Remove this multiset
            temp_M[:, next_mset] = 0
            iter += 1

        self._solution = np.array(self._solution)
        self.coverage_until_ = np.array(self.coverage_until_)
        self.n_elements_remaining_ = np.array(self.n_elements_remaining_)
        self.effective_ = np.array(self.effective_)

        solution = self._nonzero_cols[np.array(self._solution)]

        return solution.copy()

    def fit_predict(self, M, coverage, max_iters=0, multiplier=None):
        """
        Given a matrix M, construct a greedy cover instance by considering
        the columns of M as multisets. Will skip all-zero columns and
        save the "true" indices of the columns under ``self._true_indices``.
        Then returns the indices of the selected features given
        a certain coverage.

        Args:
            M: np.ndarray, the data matrix. Has shape (n_elements, n_multisets)
            coverage: int or list, in case of a list will apply a specific
                coverage to each element.
            max_iters: maximum number of iterations (features) to return.
                A value of 0 or None means no limit.
        """
        self.fit(M, multiplier)
        return self.predict(coverage, max_iters)

    @property
    def elements_incomplete_cover_(self):
        """
        Array of indices corresponding to elements that could not
        be covered to the desired coverage.
        """
        return np.where((self.max_coverage_ - self.effective_coverage_) < 0)[0]

    def _real_index_to_inner(self, i):
        """
        Converts a real index to an inner index. If the real index was
        not used to construct GCI, returns None.
        """
        check_is_fitted(self)
        if i in self._nonzero_cols:
            return (self._nonzero_cols == i).nonzero()[0][0]
        return None

    def _correct_x_multiplier(self, val):
        if isinstance(val, list):
            val = np.array(val)
        elif isinstance(val, np.ndarray):
            val = val.copy()
        if self.multiplier_ is not None:
            return val * self.multiplier_
        return val
