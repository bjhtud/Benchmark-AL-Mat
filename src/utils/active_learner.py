#!/usr/bin/env python3
"""
Active Learning Framework and Utilities

This module provides the main active learning framework and utility functions
for conducting active learning experiments on materials science datasets.

Author: [Your Name]
Email: [Your Email]
License: MIT License
Created: 2025-01-06
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm
from .initialize import initialize
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from xgboost import XGBRegressor
import time

class RandomSearch:
    """
    Random sampling baseline strategy for active learning.

    This class provides a simple random sampling strategy that can be used
    as a baseline for comparison with other active learning methods.
    """

    def __init__(self, random_state=None):
        """
        Initialize the RandomSearch strategy.

        Args:
            random_state (int, optional): Random seed for reproducibility.
        """
        self.random_state = random_state

    def query(self, X_unlabeled, n_act=1, **kwargs):
        """
        Query samples randomly from the unlabeled dataset.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data points to query from.
            n_act (int, optional): Number of samples to query. Defaults to 1.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list: Indices of randomly selected samples.
        """
        rng = np.random.default_rng(seed=int(self.random_state))
        query_idx = rng.choice(range(len(X_unlabeled)), size=n_act, replace=False)
        selected_indices = [int(i) for i in X_unlabeled.index[query_idx]]
        return selected_indices


def data_extraction(idx, X):
    """
    Extract specific samples from dataset and return remaining data.

    Args:
        idx (list): Indices of samples to extract.
        X (pd.DataFrame): Dataset to extract from.

    Returns:
        tuple: (extracted_data, remaining_data) where:
            - extracted_data: DataFrame containing extracted samples
            - remaining_data: DataFrame with extracted samples removed
    """
    # Extract specific rows using .loc
    extracted_data = X.loc[idx]
    # Remove these rows and return new DataFrame
    remaining_data = X.drop(idx)
    return extracted_data, remaining_data


def active_learning(estimators, X_t, y_t, X_val, y_val, n_initial, n_pro_query, n_queries, threshold,
                    initial_method="random", test_methods=None, random_state=36):
    """
    Main active learning loop for conducting experiments.

    This function performs the complete active learning experiment, including
    initialization, iterative query selection, and performance evaluation.

    Args:
        estimators (list): List of active learning strategies to evaluate.
        X_t (pd.DataFrame): Training feature data.
        y_t (pd.Series): Training target data.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.Series): Validation target data.
        n_initial (int): Number of initial labeled samples.
        n_pro_query (int): Number of samples to query per iteration.
        n_queries (int): Number of query iterations to perform.
        threshold (float): Performance threshold for early stopping.
        initial_method (str, optional): Method for initial sample selection.
            Defaults to "random".
        test_methods (list, optional): List of evaluation methods.
            Defaults to ["normal"].
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 36.

    Returns:
        tuple: (query_idx_all, query_time_all) where:
            - query_idx_all: Dictionary containing query indices for each strategy
            - query_time_all: Dictionary containing timing information
    """
    random_strategy = RandomSearch(random_state=random_state)
    if test_methods is None:
        test_methods = ["normal"]

    query_idx_all = {}
    query_time_all = {}

    for estimator in estimators:
        idx_batch = []
        query_time = []
        X_unlabeled = X_t.copy()
        y_unlabeled = y_t.copy()

        initial_idx = initialize(X_unlabeled, n_initial, method=initial_method)

        idx_batch.append(initial_idx)
        X_labeled, X_unlabeled = data_extraction(initial_idx, X_unlabeled)
        y_labeled, y_unlabeled = data_extraction(initial_idx, y_unlabeled)

        for _ in tqdm(range(n_queries), desc=f'{estimator.__class__.__name__} Querying', unit='query'):
            start = time.perf_counter()
            try:
                query_idx = estimator.query(X_unlabeled=X_unlabeled, n_act=n_pro_query, X_labeled=X_labeled,
                                            y_labeled=y_labeled, y_unlabeled=y_unlabeled)
            except Exception as e:
                query_idx = random_strategy.query(X_unlabeled=X_unlabeled, n_act=n_pro_query, X_labeled=X_labeled,
                                            y_labeled=y_labeled, y_unlabeled=y_unlabeled)
                
            end = time.perf_counter()
            query_time.append(end - start)

            idx_batch.append(query_idx)
            X_query, X_unlabeled = data_extraction(query_idx, X_unlabeled)
            y_query, y_unlabeled = data_extraction(query_idx, y_unlabeled)
            X_labeled = pd.concat([X_labeled, X_query])
            y_labeled = pd.concat([y_labeled, y_query])

        if estimator.__class__.__name__ == "BMDAL":
            query_idx_all[estimator.__class__.__name__ + '_' + estimator.selection_method] = idx_batch
            query_time_all[estimator.__class__.__name__ + '_' + estimator.selection_method] = query_time
        else:
            query_idx_all[estimator.__class__.__name__] = idx_batch
            query_time_all[estimator.__class__.__name__] = query_time

    return query_idx_all, query_time_all
