#!/usr/bin/env python3
"""
Greedy Sampling with Prediction Space Diversity (GSy) Active Learning Strategy

This module implements the GSy active learning strategy which uses XGBoost
regression and selects samples based on prediction space diversity to ensure
diverse outputs for active learning.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def distance(sample1, sample2):
    """
    Calculate Euclidean distance between two samples.

    Args:
        sample1 (np.ndarray): First sample vector.
        sample2 (np.ndarray): Second sample vector.

    Returns:
        float: Euclidean distance between the samples.
    """
    return np.linalg.norm(sample1 - sample2)


class GSy(XGBRegressor):
    """
    Greedy Sampling with Prediction Space Diversity (GSy) active learning strategy.

    This class extends XGBRegressor to implement an active learning strategy
    that selects samples based on diversity in the prediction space, ensuring
    that selected samples have diverse predicted outputs.

    Attributes:
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state=None):
        """
        Initialize the GSy strategy.

        Args:
            random_state (int, optional): Random seed for reproducibility.
                Defaults to None.
        """
        super().__init__(random_state=random_state)
        self.random_state = random_state

    def query(self, X_unlabeled, n_act, X_labeled, y_labeled, y_unlabeled, **kwargs):
        """
        Query samples using GSy strategy.

        Iteratively trains XGBoost model and selects samples that maximize
        the minimum distance in prediction space to already labeled samples,
        ensuring diversity in the predicted outputs.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled feature data.
            n_act (int): Number of samples to query.
            X_labeled (pd.DataFrame): Currently labeled feature data.
            y_labeled (pd.Series): Currently labeled target data.
            y_unlabeled (pd.Series): Unlabeled target data.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list: Indices of selected samples based on prediction space diversity.
        """
        XGB = XGBRegressor(random_state=self.random_state)
        selected_indices = []

        # Iteratively select samples
        while len(selected_indices) < n_act:
            # Train XGBoost model on current labeled data
            XGB.fit(X_labeled, y_labeled)

            # Predict on unlabeled data
            y_predict = XGB.predict(X_unlabeled)
            y_predict = pd.DataFrame(y_predict, index=X_unlabeled.index)

            max_distance = -1
            next_sample_index = None

            # Find sample with maximum minimum distance in prediction space
            for idx, sample in y_predict.iterrows():
                # Calculate minimum distance to all labeled target values
                min_distance = min([
                    distance(sample.values, y_labeled.loc[i].values)
                    for i in y_labeled.index
                ])

                # Select sample with maximum minimum distance (most diverse prediction)
                if min_distance > max_distance:
                    max_distance = min_distance
                    next_sample_index = idx

            # Update selected indices and datasets
            selected_indices.append(next_sample_index)
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[next_sample_index]]])
            X_unlabeled = X_unlabeled.drop(next_sample_index)
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[[next_sample_index]]])
            y_unlabeled = y_unlabeled.drop(next_sample_index)

        # Convert to integer indices
        selected_indices = [int(i) for i in selected_indices]
        return selected_indices
