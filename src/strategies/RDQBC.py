#!/usr/bin/env python3
"""
RD-QBC (Representative Diversity - Query by Committee) Active Learning Strategy

This module implements the RD-QBC active learning strategy that combines
representative diversity sampling with query by committee for uncertainty estimation.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge

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


class RD_QBC_ALR(RandomForestRegressor):
    """
    RD-QBC Active Learning Regressor.

    This class combines representative diversity sampling with query by committee
    using multiple diverse regression models to estimate prediction uncertainty.

    Attributes:
        random_state (int): Random seed for reproducibility.
        n_samples (int): Number of labeled samples.
        num_learner (int): Number of learners in the committee.
        random_searcher (RandomSearch): Fallback random searcher.
        param_grids (dict): Parameter grids for different models.
        learners (list): List of diverse regression models.
    """

    def __init__(self, random_state=None, num_learner=10):
        """
        Initialize RD_QBC_ALR strategy.

        Args:
            random_state (int, optional): Random seed. Defaults to None.
            num_learner (int, optional): Number of learners in committee. Defaults to 10.
        """
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.n_samples = 0
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.num_learner = num_learner
        self.random_searcher = RandomSearch(random_state=random_state)
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
        models = list(self.param_grids.keys())
        base_count = self.num_learner // len(models)  # 每种回归器基础分配数
        remainder = self.num_learner % len(models)
        model_counts = {model: base_count for model in models}
        extra_models = random.sample(models, remainder)
        for model in extra_models:
            model_counts[model] += 1

        # 初始化学习器，每种学习器都定义不同数量的估计器
        self.learners = []

        for model_name, count in model_counts.items():
            param_grid = list(ParameterGrid(self.param_grids[model_name]))  # 获取超参数网格
            random.shuffle(param_grid)  # 随机打乱网格，确保多样性
            for i in range(count):
                params = param_grid[i % len(param_grid)]  # 选取不同超参数组合
                if model_name == "XGBRegressor":
                    model = XGBRegressor(random_state=self.random_state, **params)
                elif model_name == "MLPRegressor":
                    model = MLPRegressor(random_state=self.random_state, **params)
                # elif model_name == "SVR":
                #     model = SVR(**params)
                # elif model_name == "GaussianProcessRegressor":
                #     model = GaussianProcessRegressor(**params)
                elif model_name == "KNeighborsRegressor":
                    model = KNeighborsRegressor(**params)
                elif model_name == "BayesianRidge":
                    model = BayesianRidge(**params)
                self.learners.append(model)

    def query(self, X_unlabeled, X_labeled, y_labeled, y_unlabeled, n_act=1):
        """
        Query samples using RD-QBC strategy.

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
            cluster_indices = np.where(labels == largest_cluster)[0]
            center = centers[largest_cluster]
            cluster_points = X.iloc[cluster_indices]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_point_index = X.index[cluster_indices[np.argmin(distances)]]

            # QBC 选择
            committee_predictions = []
            for j in range(self.num_learner):
                # 随机抽样
                bootstrap_indices = np.random.choice(labeled_indices, size=len(labeled_indices), replace=True)
                model = self.learners[j]
                model.fit(X_labeled.loc[bootstrap_indices], y_labeled.loc[bootstrap_indices])

                predictions = model.predict(cluster_points)
                predictions = predictions.reshape(-1, 1)
                committee_predictions.append(predictions)

            committee_predictions = np.array(committee_predictions)
            variances = np.var(committee_predictions, axis=0)
            most_uncertain_index = cluster_indices[np.argmax(variances)]

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
