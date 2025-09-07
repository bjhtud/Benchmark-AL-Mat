#!/usr/bin/env python3
"""
Query by Diversity and Density (QDD) Active Learning Strategy

This module implements the QDD active learning strategy which combines uncertainty,
diversity, and density measures to select the most informative samples for labeling
in regression tasks using Random Forest ensemble methods.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor


class QDD(RandomForestRegressor):
    """
    Query by Diversity and Density (QDD) active learning strategy.

    This class extends RandomForestRegressor to implement an active learning strategy
    that combines uncertainty (prediction variance), diversity (distance to labeled samples),
    and density (local neighborhood density) to select the most informative samples.

    Attributes:
        a (float): Weight parameter for diversity component.
        b (float): Weight parameter for density component.
        The uncertainty weight is automatically calculated as (1 - a - b).
    """

    def __init__(self, a=0.333, b=0.333, random_state=None):
        """
        Initialize the QDD strategy.

        Args:
            a (float, optional): Weight for diversity component. Defaults to 0.333.
            b (float, optional): Weight for density component. Defaults to 0.333.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to None.

        Raises:
            ValueError: If a + b > 1 or a + b < 0.
        """
        self.a = a
        self.b = b

        # Ensure a and b sum is valid (uncertainty weight = 1 - a - b >= 0)
        if a + b > 1 or a + b < 0:
            raise ValueError('a + b should be between 0 and 1 to ensure valid uncertainty weight')

        super().__init__(random_state=random_state)

    def calculate_knn_density(self, X_unlabeled, k):
        """
        Calculate KNN-based density for each unlabeled sample.

        Computes local density using k-nearest neighbors and cosine similarity.
        Higher density indicates the sample is in a dense region of the feature space.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled sample features.
            k (int): Number of nearest neighbors to consider.

        Returns:
            np.ndarray: Density values for each unlabeled sample.
        """
        # Train K-nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='cosine').fit(X_unlabeled)
        _, indices = nbrs.kneighbors(X_unlabeled)

        # Calculate KNN-density for each point
        densities = []
        for i in range(X_unlabeled.shape[0]):
            # Get k nearest neighbor indices (excluding self, so start from index 1)
            neighbors = indices[i][1:]

            # Calculate cosine similarities (1 - cosine distance)
            cosine_similarities = [
                1 - cosine(X_unlabeled.iloc[i], X_unlabeled.iloc[neighbor])
                for neighbor in neighbors
            ]
            knn_density = np.mean(cosine_similarities)
            densities.append(knn_density)

        return np.array(densities)

    def calculate_distance(self, X_unlabeled, X_labeled):
        """
        Calculate minimum distance from unlabeled samples to labeled samples.

        Computes the Euclidean distance from each unlabeled sample to its nearest
        labeled sample. Higher distances indicate more diverse samples.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled sample features.
            X_labeled (pd.DataFrame): Labeled sample features.

        Returns:
            np.ndarray: Minimum distances from unlabeled to labeled samples.
        """
        distances = distance.cdist(X_unlabeled, X_labeled, 'euclidean')
        distances = np.min(distances, axis=1)
        return distances

    def query(self, X_unlabeled, n_act, X_labeled, y_labeled, y_unlabeled, **kwargs):
        """
        Query samples using QDD strategy.

        Iteratively selects samples that optimize a combination of uncertainty
        (prediction variance), diversity (distance to labeled samples), and
        density (local neighborhood density).

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled feature data.
            n_act (int): Number of samples to query.
            X_labeled (pd.DataFrame): Currently labeled feature data.
            y_labeled (pd.Series): Currently labeled target data.
            y_unlabeled (pd.Series): Unlabeled target data.
            **kwargs: Additional keyword arguments.

        Returns:
            list: Indices of selected samples based on QDD criteria.
        """
        selected_indices = []

        # Iteratively select samples
        for i in range(n_act):
            # Train Random Forest on current labeled data
            self.fit(X_labeled, y_labeled)

            # Calculate prediction variance (uncertainty)
            predictions = np.array([tree.predict(X_unlabeled) for tree in self.estimators_])
            variance = np.var(predictions, axis=0)

            # Calculate KNN-density
            k = max(1, X_unlabeled.shape[0] // 5)  # Use 20% of samples as k
            densities = self.calculate_knn_density(X_unlabeled, k)

            # Calculate distance to labeled samples (diversity)
            distances = self.calculate_distance(X_unlabeled, X_labeled)

            # Combine uncertainty, diversity, and density
            uncertainty_component = (1 - self.a - self.b) * variance
            diversity_component = self.a * distances
            density_component = self.b * densities

            # Final uncertainty measure combining all components
            combined_score = uncertainty_component + diversity_component + density_component

            # Select sample with maximum combined score
            selected_index = np.argmax(combined_score)
            original_index = X_unlabeled.index[selected_index]
            selected_indices.append(original_index)

            # Add selected sample to labeled set
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[original_index]]], axis=0)
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[[original_index]]], axis=0)

            # Remove selected sample from unlabeled set
            X_unlabeled = X_unlabeled.drop(index=original_index)
            y_unlabeled = y_unlabeled.drop(index=original_index)

        # Convert to integer indices
        selected_indices = [int(i) for i in selected_indices]
        return selected_indices
