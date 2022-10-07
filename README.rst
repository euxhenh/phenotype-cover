This repository provides two algorithms for the phenotype cover (PC)
biomarker selection problem. GreedyPC is based on the extended greedy
algorithm to set cover, and CEM-PC is based on the cross-entropy-method.

Install via

    pip install multiset-multicover
    pip install phenotype-cover

Other packages that phenotype-cover depends on are numpy, matplotlib, and scikit-learn.

Import `GreedyPC` or `CEMPC` from `phenotype_cover`.

Example

    >>> from phenotype_cover import GreedyPC

    >>> # Multiply data matrix by 100 to avoid a zero-matrix after the "rounding" step
    >>> X = np.random.random((1000, 200)) * 100  # data matrix
    >>> y = np.random.randint(0, 5, 1000)  # class labels

    >>> gpc = GreedyPC()
    >>> gpc.fit(X, y)
    >>> features = gpc.select(10)  # coverage of 10

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
