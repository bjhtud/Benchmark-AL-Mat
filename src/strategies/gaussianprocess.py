#!/usr/bin/env python3
"""
Gaussian Process Based Active Learning Strategy

This module implements an uncertainty-based active learning strategy using
Gaussian Process regression to select the most informative samples.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor


class GaussianProcessBased(GaussianProcessRegressor):
    """
    Gaussian Process based active learning strategy.

    This class implements an uncertainty sampling strategy using Gaussian Process
    regression, selecting samples with the highest prediction uncertainty.

    Inherits from sklearn's GaussianProcessRegressor and adds query functionality
    for active learning.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GaussianProcessBased strategy.

        Args:
            **kwargs: Keyword arguments passed to GaussianProcessRegressor.
        """
        super().__init__(**kwargs)

    def query(self, X_unlabeled, n_act=1, **kwargs):
        """
        Query samples with highest prediction uncertainty.

        Selects samples where the Gaussian Process has the highest prediction
        uncertainty (standard deviation).

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data points to query from.
            n_act (int, optional): Number of samples to query. Defaults to 1.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list: Indices of selected samples with highest uncertainty.
        """
        # Get predictions with uncertainty estimates
        y_hat, std = self.predict(X_unlabeled, return_std=True)

        query_idx = np.argsort(std)[-n_act:]
        selected_indices = [i for i in X_unlabeled.index[query_idx.tolist()]]
        return selected_indices
