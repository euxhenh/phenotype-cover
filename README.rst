This repository provides two algorithms for the phenotype cover (PC)
biomarker selection problem introduced in the paper:
"`Multiset multicover methods for discriminative marker selection <https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(22)00229-6>`_".
GreedyPC is based on the extended greedy algorithm to set cover, and CEM-PC is based on the cross-entropy-method.

Install via

.. code:: bash

    pip install multiset-multicover
    pip install phenotype-cover

Other packages that phenotype-cover depends on are numpy, matplotlib, and scikit-learn.

Import `GreedyPC` or `CEMPC` from `phenotype_cover`.

Example

    >>> from phenotype_cover import GreedyPC
    >>> from sklearn.datasets import make_classification

    >>> # You may need to log-transform X if working with raw counts
    >>> X, y = make_classification(1000, 200, n_informative=5, n_classes=5, scale=100)

    >>> gpc = GreedyPC()
    >>> gpc.fit(X, y)
    >>> features = gpc.select(100)  # coverage of 100

Some other functionality implemented in GreedyPC

    >>> # Number of elements reamining and coverage attained after every iteration
    >>> gpc.plot_progress()
    >>> gpc.n_elements_remaining_per_iter_
    >>> gpc.coverage_per_iter_

    >>> # Heatmap of the coverage provided by some feature i
    >>> gpc.feature_coverage(i)

    >>> # Maximum possible coverage for evey class pair
    >>> gpc.max_coverage()

    >>> # Pairs that could not be covered to the desired `coverage`
    >>> gpc.pairs_with_incomplete_cover_
