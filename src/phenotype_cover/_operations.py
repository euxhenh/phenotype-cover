from itertools import combinations

import numpy as np
from sklearn.utils.validation import (as_float_array, assert_all_finite,
                                      check_X_y, column_or_1d, indexable)


def group_by(X, y, *, category_orders=None, operation=lambda x: x.mean(axis=0)):
    """Groups the samples in X by labels in y and applies `operation`
    to the aggregated groups.

    Parameters
    __________
    X: array-like of shape (n_samples, n_features)
        The data matrix.
    y: array-like of shape (n_samples,)
        The class labels.
    category_orders: array-like of shape (np.unique(y).size,)
        Order of class labels to use when constructing the matrix.
        If None, will sort the class labels alphabetically.
    operation: callable
        The function to apply to the aggregated groups.
    """
    X, y = check_X_y(X, y, accept_sparse=["csr"])
    X = indexable(X)[0]

    if category_orders is None:
        category_orders = np.unique(y)
    elif not set(category_orders).issubset(set(y)):
        # To avoid getting nan values
        raise ValueError("Found categories not present in `y`.")
    else:
        category_orders = column_or_1d(category_orders)

    if not callable(operation):
        raise ValueError("Please pass a callable operation.")

    M = np.zeros((len(category_orders), X.shape[1]))

    for i, category in enumerate(category_orders):
        _agg_values = operation(X[y == category])
        _agg_values = as_float_array(_agg_values).flatten()
        if len(_agg_values) != X.shape[1]:
            raise ValueError(
                "Operation must return a vector of size X.shape[1]"
                f"but instead found vector of size {len(_agg_values)}."
            )
        assert_all_finite(_agg_values)
        M[i] = _agg_values

    return M


def pairwise_differences(
        X, y,
        *,
        classes=None,
        ordered=False,
        operation=lambda x: x.mean(axis=0)):
    """
    Given an data matrix X, if ordered is False, construct a matrix P of shape
    (n * (n-1) / 2, X.shape[1]) where n is the number of classes in y.
    The (i*j, g) entry of P corresponds to the average expression of feature g
    in group i - average expression of feature g in group j, in absolute value.
    If ordered is True, the shape of P will be (n * (n-1), X.shape[1]) and
    the pairwise distances will be clipped at 0.

    Returns P and a dictionary of mappings: label, label -> index.

    Parameters
    _________
    X: np.ndarray of shape (n_samples, n_features)
    y: np.ndarray of shape (n_samples,)
    classes: np.ndarray or None, unique class labels in y
    ordered: bool, if True will construct a matrix of ordered
        pairwise differences. In this case the shape of P is
        (n * (n-1), X.shape[1]).
    operation: callable, operation to use when constructing the class vector.
    """
    if classes is None:
        classes = np.unique(y)

    n_classes = len(classes)
    # All pairwise combinations
    n_class_pairs = n_classes * (n_classes - 1) // 2

    # Cache the average vector of each class
    class_averages = group_by(
        X, y, category_orders=classes, operation=operation)

    # Compute the actual pairwise differences
    P = np.zeros((n_class_pairs * (1 if not ordered else 2), X.shape[1]))
    index_to_pair_dict = {}

    # Make sure to use range(n_classes) when indexing instead of classes,
    # to allow for arbitrary class labels.
    for index, (i, j) in enumerate(combinations(range(n_classes), 2)):
        difference = class_averages[i] - class_averages[j]
        if ordered:
            # Clip negative values to 0
            # Assign i - j to index and j - i to index + n_class_pairs
            P[index] = np.clip(difference, 0, None)
            index_to_pair_dict[index] = (i, j)
            P[index + n_class_pairs] = np.clip(-difference, 0, None)
            index_to_pair_dict[index + n_class_pairs] = (j, i)
        else:
            P[index] = np.abs(difference)
            index_to_pair_dict[index] = (i, j)

    return P, index_to_pair_dict
