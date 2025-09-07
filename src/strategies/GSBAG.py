#!/usr/bin/env python3
"""
Gaussian Process with Bayesian Active Learning by Greedy Selection (GSBAG)

This module implements the GSBAG active learning strategy which combines
Gaussian Process regression with greedy selection based on posterior variance
maximization for optimal sample acquisition.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C


class GSBAG(GaussianProcessRegressor):
    """
    Gaussian Process with Bayesian Active Learning by Greedy Selection.

    This class extends GaussianProcessRegressor to implement an active learning
    strategy that greedily selects samples to maximize the posterior variance,
    which corresponds to the uncertainty in the Gaussian Process predictions.

    Attributes:
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state=None, kernel=None, **kwargs):
        """
        Initialize the GSBAG strategy.

        Args:
            random_state (int, optional): Random seed for reproducibility.
                Defaults to None.
            kernel: Gaussian Process kernel function. Defaults to None.
            **kwargs: Additional keyword arguments passed to GaussianProcessRegressor.
        """
        super().__init__(kernel=kernel, random_state=random_state, **kwargs)
        self.random_state = random_state

    def query(self, X_unlabeled, n_act, X_labeled, y_labeled, y_unlabeled, **kwargs):
        """
        Query samples using GSBAG strategy.

        Selects samples by greedily maximizing the posterior variance of the
        Gaussian Process. This approach iteratively selects the sample that
        would provide the maximum information gain.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled feature data.
            n_act (int): Number of samples to query.
            X_labeled (pd.DataFrame): Currently labeled feature data.
            y_labeled (pd.Series): Currently labeled target data.
            y_unlabeled: Unlabeled target data (unused).
            **kwargs: Additional keyword arguments.

        Returns:
            list: Indices of selected samples that maximize posterior variance.

        Raises:
            ValueError: If the Gaussian Process has not been fitted yet.
        """
        if X_labeled is None:
            raise ValueError("You have to fit the Gaussian Process at least once first")

        # Fit the Gaussian Process on current labeled data
        self.fit(X_labeled, y_labeled)

        selected_X = pd.DataFrame(columns=X_unlabeled.columns)
        selected_indices = []

        # Extract noise variance from kernel
        sigma2_epsilon = self.kernel_.k2.noise_level  # Adjust based on kernel structure

        # Iteratively select samples that maximize posterior variance
        for i in range(n_act):
            # Combine labeled data with previously selected samples
            combined_X = pd.concat([X_labeled, selected_X])

            # Compute kernel matrix and its inverse
            K = self.kernel_(combined_X.values.astype(float), combined_X.values.astype(float))
            K_inv = np.linalg.inv(K + np.eye(K.shape[0]) * sigma2_epsilon)

            pi_star_values = []

            # Evaluate posterior variance for each unlabeled sample
            for idx, x in enumerate(X_unlabeled.values.astype(float)):
                x = x.reshape(1, -1)
                k_x = self.kernel_(combined_X.values.astype(float), x)
                k_xx = self.kernel_(x, x)

                # Calculate posterior variance (information gain metric)
                pi_star = k_xx - k_x.T @ K_inv @ k_x
                pi_star_values.append(pi_star.squeeze())

            pi_star_values = np.array(pi_star_values)

            # Select sample with maximum posterior variance
            selected_idx = np.argmax(pi_star_values)
            selected_induce = X_unlabeled.index[selected_idx]

            # Update selected samples and remove from unlabeled pool
            selected_X = pd.concat([selected_X, X_unlabeled.loc[[selected_induce]]])
            X_unlabeled = X_unlabeled.drop(selected_induce)
            selected_indices.append(selected_induce)

        # Convert to integer indices
        selected_indices = [int(i) for i in selected_indices]
        return selected_indices
