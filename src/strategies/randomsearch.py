#!/usr/bin/env python3
"""
Random Search Active Learning Strategy

This module implements a random sampling strategy for active learning,
which serves as a baseline for comparison with other query strategies.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor


class RandomSearch(RandomForestRegressor):
    """
    Random Search active learning strategy.

    This class implements a simple random sampling strategy for active learning,
    inheriting from RandomForestRegressor but using random selection for queries.

    Attributes:
        random_state (int, optional): Random state for reproducibility.
    """

    def __init__(self, random_state=None):
        """
        Initialize the RandomSearch strategy.

        Args:
            random_state (int, optional): Random seed for reproducible results.
                Defaults to None.
        """
        super(RandomSearch, self).__init__()
        self.random_state = random_state

    def query(self, X_unlabeled, n_act=1, **kwargs):
        """
        Query samples randomly from the unlabeled dataset.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data points to query from.
            n_act (int, optional): Number of samples to query. Defaults to 1.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list: Indices of selected samples.
        """
        rng = np.random.default_rng(seed=int(self.random_state))
        query_idx = rng.choice(range(len(X_unlabeled)), size=n_act, replace=False)
        selected_indices = [int(i) for i in X_unlabeled.index[query_idx]]
        return selected_indices
