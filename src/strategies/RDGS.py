#!/usr/bin/env python3
"""
RD-GS (Representative Diversity - Greedy Selection) Active Learning Strategy

This module implements the RD-GS active learning strategy that combines
representative diversity sampling with greedy selection for active learning.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def data_extraction(idx, X):
    """
    Extract specific indices from dataset.

    Args:
        idx: Indices to extract.
        X (pd.DataFrame): Input dataset.

    Returns:
        tuple: Extracted data and remaining data.
    """
    # 使用 .loc 提取特定索引的行
    extracted_data = X.loc[idx]
    # 使用 .drop 删除这些行，注意设置 inplace=False 以返回新的 DataFrame
    remaining_data = X.drop(idx)
    return extracted_data, remaining_data


class RandomSearch:
    """
    Random Search strategy for fallback selection.

    Attributes:
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state=None):
        """
        Initialize RandomSearch strategy.

        Args:
            random_state (int, optional): Random seed. Defaults to None.
        """
        self.random_state = random_state

    def query(self, X_unlabeled, n_act=1, **kwargs):
        """
        Query samples randomly.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data.
            n_act (int, optional): Number of samples to select. Defaults to 1.
            **kwargs: Additional arguments.

        Returns:
            list: Selected sample indices.
        """
        rng = np.random.default_rng(seed=int(self.random_state))
        query_idx = rng.choice(range(len(X_unlabeled)), size=n_act, replace=False)
        selected_indices = [int(i) for i in X_unlabeled.index[query_idx]]
        return selected_indices


class RD_GS_ALR(RandomForestRegressor):
    """
    RD-GS Active Learning Regressor.

    This class combines representative diversity sampling with greedy selection
    to choose samples that are both representative and maximally distant from
    labeled samples.

    Attributes:
        random_state (int): Random seed for reproducibility.
        n_samples (int): Number of labeled samples.
        random_searcher (RandomSearch): Fallback random searcher.
    """

    def __init__(self, random_state=None):
        """
        Initialize RD_GS_ALR strategy.

        Args:
            random_state (int, optional): Random seed. Defaults to None.
        """
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.n_samples = 0
        self.random_searcher = RandomSearch(random_state=random_state)

    def query(self, X_unlabeled, X_labeled, n_act=1, **kwargs):
        """
        Query samples using RD-GS strategy.

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
            # RD 部分
            kmeans = KMeans(n_clusters=self.n_samples + i, random_state=self.random_state)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            labeled_indices = X_labeled.index
            labeled_labels = [labels[X.index.get_loc(idx)] for idx in labeled_indices]

            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))

            excluded_clusters = np.unique(labeled_labels)
            potential_clusters = [c for c in cluster_sizes.keys() if c not in excluded_clusters]

            if not potential_clusters:
                continue

            largest_cluster = max(potential_clusters, key=lambda x: cluster_sizes[x])
            cluster_indices = X.index[labels == largest_cluster]

            # GS 部分
            # 计算已标记样本到簇中心点的最小距离
            min_dist = -np.inf
            closest_point_index = None
            for idx in cluster_indices:
                point = X_unlabeled.loc[idx]
                # 计算该点到所有已标记样本的距离
                dists_to_labeled = np.linalg.norm(X_labeled - point, axis=1)
                min_dist_to_labeled = np.min(dists_to_labeled)

                if min_dist_to_labeled > min_dist:
                    min_dist = min_dist_to_labeled
                    closest_point_index = idx

            if closest_point_index in X_unlabeled.index:
                selected_samples.append(closest_point_index)
                X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[closest_point_index]]])
                X_unlabeled = X_unlabeled.drop(closest_point_index)

        selected_samples = [int(i) for i in selected_samples]
        if len(selected_samples) < n_act:
            query_idx = self.random_searcher.query(X_unlabeled, n_act - len(selected_samples))
            selected_samples += query_idx
        return selected_samples
