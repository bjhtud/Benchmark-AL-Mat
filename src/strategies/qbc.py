#!/usr/bin/env python3
"""
Query by Committee Active Learning Strategy

This module implements the Query by Committee (QBC) active learning strategy
using an ensemble of diverse regressors with different hyperparameters.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import random
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score


class QueryByCommittee:
    """
    Query by Committee active learning strategy.

    This class implements the QBC strategy using an ensemble of diverse regression
    models with different hyperparameters. It selects samples where the ensemble
    predictions have the highest disagreement (standard deviation).

    Attributes:
        random_state (int, optional): Random state for reproducibility.
        num_learner (int): Number of learners in the committee.
        param_grids (dict): Parameter grids for different model types.
        learners (list): List of initialized regression models.
    """

    def __init__(self, random_state=None, num_learner=30):
        """
        Initialize the Query by Committee strategy.

        Args:
            random_state (int, optional): Random seed for reproducible results.
                Defaults to None.
            num_learner (int, optional): Number of learners in the committee.
                Defaults to 30.
        """
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.num_learner = num_learner

        # Define parameter grids for different regression models
        self.param_grids = {
            "XGBRegressor": {
                "n_estimators": [50, 100, 150, 200, 250, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
                "max_depth": [3, 5, 7, 9, 11, 13],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0, 0.5],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0, 0.5],
                "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "MLPRegressor": {
                "hidden_layer_sizes": [(50, 10), (25, 5), (25, 50, 10), (10, 25, 5), (25, 50, 50, 10), (50, 100, 100, 20)],
                "alpha": [0.0001, 0.001, 0.01, 0.00005, 0.005, 0.00001],
                "learning_rate_init": [0.001, 0.005, 0.01, 0.02, 0.03, 0.005],
                "activation": ["relu", "tanh", "logistic", "identity", "relu", "tanh"],
                "max_iter": [200, 300, 400, 500, 600, 1000]
            },
            "KNeighborsRegressor": {
                "n_neighbors": [3, 4, 5, 6, 7, 8],
                "weights": ["uniform", "distance", "uniform", "distance", "uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute", "auto", "ball_tree"],
                "leaf_size": [30, 20, 40, 10, 50, 60],
                "p": [1, 2, 3, 4, 5, 1],
                "metric": ["minkowski", "euclidean", "manhattan", "chebyshev", "minkowski", "euclidean"]
            },
            "BayesianRidge": {
                "alpha_1": [1e-6, 1e-7, 1e-5, 1e-4, 1e-3, 1e-8],
                "alpha_2": [1e-6, 1e-7, 1e-5, 1e-4, 1e-3, 1e-8],
                "lambda_1": [1e-6, 1e-7, 1e-5, 1e-4, 1e-3, 1e-8],
                "lambda_2": [1e-6, 1e-7, 1e-5, 1e-4, 1e-3, 1e-8],
            }
        }

        # Initialize learners with diverse hyperparameters
        self._initialize_learners()

    def _initialize_learners(self):
        """
        Initialize learners with diverse hyperparameters.

        Creates a committee of regression models with different types and
        hyperparameter configurations to ensure diversity.
        """
        models = list(self.param_grids.keys())
        base_count = self.num_learner // len(models)  # Base allocation per model type
        remainder = self.num_learner % len(models)
        model_counts = {model: base_count for model in models}

        # Distribute remaining learners randomly
        extra_models = random.sample(models, remainder)
        for model in extra_models:
            model_counts[model] += 1

        # Initialize learners with different hyperparameter combinations
        self.learners = []

        for model_name, count in model_counts.items():
            param_grid = list(ParameterGrid(self.param_grids[model_name]))
            random.shuffle(param_grid)  # Shuffle to ensure diversity

            for i in range(count):
                params = param_grid[i % len(param_grid)]  # Cycle through parameter combinations

                if model_name == "XGBRegressor":
                    model = XGBRegressor(random_state=self.random_state, **params)
                elif model_name == "MLPRegressor":
                    model = MLPRegressor(random_state=self.random_state, **params)
                elif model_name == "KNeighborsRegressor":
                    model = KNeighborsRegressor(**params)
                elif model_name == "BayesianRidge":
                    model = BayesianRidge(**params)

                self.learners.append(model)

    def fit(self, X, y):
        """
        Fit method for compatibility (not used in active learning).

        Args:
            X: Feature matrix.
            y: Target vector.
        """
        pass

    def fit_for_test(self, X, y):
        """
        Fit all learners in the committee for testing purposes.

        Args:
            X: Training feature matrix.
            y: Training target vector.
        """
        for learner in self.learners:
            learner.fit(X, y)

    def test_for_test(self, X, y, X_test, y_test):
        """
        Test the committee on test data and return R2 scores.

        Args:
            X: Training feature matrix.
            y: Training target vector.
            X_test: Test feature matrix.
            y_test: Test target vector.

        Returns:
            list: R2 scores for each learner in the committee.
        """
        self.fit_for_test(X, y)

        predictions = np.array(
            [learner.predict(X_test).reshape((-1, y_test.shape[1])) for learner in self.learners]).reshape(
            (len(self.learners), -1))

        R2_scores = [r2_score(y_test, prediction) for prediction in predictions]
        return R2_scores

    def query(self, X_unlabeled, n_act, X_labeled, y_labeled, y_unlabeled):
        """
        Query samples where the committee has the highest disagreement.

        Uses bootstrap sampling to train each learner on different subsets
        of the labeled data, then selects samples where predictions have
        the highest standard deviation.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data points to query from.
            n_act (int): Number of samples to query.
            X_labeled (pd.DataFrame): Currently labeled feature data.
            y_labeled (pd.Series): Currently labeled target data.
            y_unlabeled: Unlabeled target data (unused).

        Returns:
            list: Indices of selected samples with highest disagreement.
        """
        # Create bootstrap samples for each learner
        datasets_X = []
        datasets_y = []

        for i in range(self.num_learner):
            datasets_X.append(X_labeled.sample(frac=1, replace=True))

        for i in range(self.num_learner):
            datasets_y.append(y_labeled.loc[datasets_X[i].index])

        for i in range(self.num_learner):
            self.learners[i].fit(datasets_X[i], datasets_y[i])

        # Get predictions from all learners
        predictions = np.array(
            [learner.predict(X_unlabeled).reshape((-1, y_labeled.shape[1])) for learner in self.learners]).reshape(
            (len(self.learners), -1))

        # Calculate standard deviation across predictions (committee disagreement)
        std_dev = np.std(predictions, axis=0)
        query_idx = np.argsort(std_dev)[-n_act:]
        selected_indices = [int(i) for i in X_unlabeled.index[query_idx.tolist()]]
        return selected_indices
