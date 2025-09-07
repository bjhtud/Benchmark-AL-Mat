#!/usr/bin/env python3
"""
RD-ALR (Representative Diversity - Active Learning Regressor) Strategy

This module implements the RD-ALR active learning strategy that uses clustering
to find representative samples for active learning in regression tasks.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Basic_RD_ALR(RandomForestRegressor):
    """
    Basic Representative Diversity Active Learning Regressor.

    This class implements an active learning strategy that uses K-means clustering
    to select representative samples from the largest clusters.

    Attributes:
        random_state (int): Random seed for reproducibility.
        n_samples (int): Number of labeled samples.
    """

    def __init__(self, random_state=None):
        """
        Initialize the Basic_RD_ALR strategy.

        Args:
            random_state (int, optional): Random seed for reproducible results.
                Defaults to None.
        """
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.n_samples = 0

    def query(self, X_unlabeled, X_labeled, n_act=1, **kwargs):
        """
        Query samples using representative diversity strategy.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data features.
            X_labeled (pd.DataFrame): Labeled data features.
            n_act (int, optional): Number of samples to select. Defaults to 1.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list: Indices of selected samples.
        """
        self.n_samples = len(X_labeled)
        X = pd.concat([X_labeled, X_unlabeled])
        selected_samples = []

        for i in range(1, n_act + 1):
            kmeans = KMeans(n_clusters=self.n_samples + i, random_state=self.random_state)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            # Correctly mapping indices to labels
            labeled_indices = X_labeled.index  # 已标记数据集的索引
            labeled_labels = [labels[X.index.get_loc(idx)] for idx in labeled_indices]  # 已标记数据集的标签

            # Find the largest cluster that doesn't include labeled indices
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))

            # Exclude clusters that contain labeled samples
            excluded_clusters = np.unique(labeled_labels)
            potential_clusters = [c for c in cluster_sizes.keys() if c not in excluded_clusters]

            # Find the largest valid cluster
            largest_cluster = max(potential_clusters, key=lambda x: cluster_sizes[x])
            cluster_indices = np.where(labels == largest_cluster)[0]
            # Find the point closest to the center of this cluster
            center = centers[largest_cluster]
            cluster_points = X.iloc[cluster_indices]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_point_index = X.index[cluster_indices[np.argmin(distances)]]
            # Ensure the selected point is from unlabeled data
            if closest_point_index in X_unlabeled.index:
                selected_samples.append(closest_point_index)

            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[closest_point_index]]])
            X_unlabeled = X_unlabeled.drop(closest_point_index)

        selected_samples = [int(i) for i in selected_samples]
        return selected_samples
