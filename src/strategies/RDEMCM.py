#!/usr/bin/env python3
"""
RD-EMCM (Representative Diversity - Expected Model Change Maximization) Active Learning Strategy

This module implements the RD-EMCM active learning strategy that combines
representative diversity sampling with expected model change maximization.

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
from xgboost import XGBRegressor

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
    # 使用 .iloc 提取特定索引的行
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


class RD_EMCM_ALR(RandomForestRegressor):
    """
    RD-EMCM Active Learning Regressor.

    This class combines representative diversity sampling with expected model
    change maximization for active learning in regression tasks.

    Attributes:
        random_state (int): Random seed for reproducibility.
        n_boots (int): Number of bootstrap samples for ensemble.
        random_searcher (RandomSearch): Fallback random searcher.
    """

    def __init__(self, random_state=None, n_boots=10):
        """
        Initialize RD_EMCM_ALR strategy.

        Args:
            random_state (int, optional): Random seed. Defaults to None.
            n_boots (int, optional): Number of bootstrap samples. Defaults to 10.
        """
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.n_samples = 0
        self.n_boots = n_boots
        self.random_searcher = RandomSearch(random_state=random_state)

    def query(self, X_unlabeled, X_labeled, y_labeled, y_unlabeled, n_act=1):
        """
        Query samples using RD-EMCM strategy.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data features.
            X_labeled (pd.DataFrame): Labeled data features.
            y_labeled (pd.Series): Labeled data targets.
            y_unlabeled (pd.Series): Unlabeled data targets.
            n_act (int, optional): Number of samples to select. Defaults to 1.

        Returns:
            list: Indices of selected samples.
        """
        self.n_samples = len(X_labeled)
        X = pd.concat([X_labeled, X_unlabeled])
        selected_samples = []

        for i in range(1, n_act + 1):
            kmeans = KMeans(n_clusters=self.n_samples + i, random_state=self.random_state)
            kmeans.fit(X)
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
            cluster_indices = np.where(labels == largest_cluster)[0]

            cluster_points = X.iloc[cluster_indices]

            # EMCM 选择
            estimator_initial = XGBRegressor(random_state=self.random_state)
            estimator_initial.fit(X_labeled, y_labeled)
            initial_predictions = estimator_initial.predict(cluster_points)
            initial_predictions = initial_predictions.reshape(1, -1)

            committee_predictions = []
            for j in range(self.n_boots):
                # 随机抽样
                bootstrap_indices = np.random.choice(labeled_indices, size=len(labeled_indices), replace=True)
                model = XGBRegressor(random_state=self.random_state)
                model.fit(X_labeled.loc[bootstrap_indices], y_labeled.loc[bootstrap_indices])

                predictions = model.predict(cluster_points)
                committee_predictions.append(predictions)

            committee_predictions = np.array(committee_predictions)
            committee_predictions.reshape(self.n_boots, -1)

            g = np.mean(committee_predictions - initial_predictions, axis=0)

            most_uncertain_index = cluster_indices[np.argmax(g)]

            # 确保选择的点来自未标记数据
            if X.index[most_uncertain_index] in X_unlabeled.index:
                selected_samples.append(X.index[most_uncertain_index])
                X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[X.index[most_uncertain_index]]]])
                y_labeled = pd.concat([y_labeled, y_unlabeled.loc[[X.index[most_uncertain_index]]]])
                X_unlabeled = X_unlabeled.drop(X.index[most_uncertain_index])
                y_unlabeled = y_unlabeled.drop(X.index[most_uncertain_index])

        selected_samples = [int(i) for i in selected_samples]
        if len(selected_samples) < n_act:
            query_idx = self.random_searcher.query(X_unlabeled, n_act - len(selected_samples))

            selected_samples += query_idx
        return selected_samples