#!/usr/bin/env python3
"""
EMCM (Expected Model Change Maximization) Active Learning Strategy

This module implements the EMCM active learning strategy using ensemble methods
and gradient boosting for uncertainty estimation in regression tasks.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np


def initialize(data, labels, Z):
    """
    Initialize bootstrap ensemble for EMCM.

    Args:
        data (array-like): Training data features.
        labels (array-like): Training data labels.
        Z (int): Number of bootstrap samples.

    Returns:
        tuple: Empty batch list and list of trained GBDT models.
    """
    n_samples = len(data)
    # 初始化批次列表
    b = []
    # 初始化引导样本集
    B_Z = []
    for _ in range(Z):
        # 从数据集中随机采样，替换采样创建引导样本
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_data, sample_labels = data[indices], labels[indices]
        # 训练一个GBDT模型
        model = GradientBoostingRegressor()
        model.fit(sample_data, sample_labels)
        B_Z.append(model)

    return b, B_Z


def select_samples(data, labels, U_indices, b_indices, B_Z, k):
    """
    Select samples using EMCM strategy.

    Args:
        data (pd.DataFrame): Feature data.
        labels (pd.Series): Label data.
        U_indices (list): Unlabeled sample indices.
        b_indices (list): Selected batch indices.
        B_Z (list): List of trained models.
        k (int): Number of samples to select per iteration.

    Returns:
        list: Updated batch indices with selected samples.
    """
    # U_indices 是未标记数据的索引列表
    # b_indices 是已选择的批次索引列表
    # data 和 labels 是特征和标签的 DataFrame
    # B_Z 是引导样本集中的模型列表
    # k 是每次迭代要选择的样本数量

    for _ in range(k):
        max_change = -np.inf
        selected_index = None
        selected_model_change = None

        # 遍历未标记的数据池
        for idx in U_indices:
            model_change = 0

            # 计算每个模型对当前样本的改变量
            for model in B_Z:
                # 使用 DataFrame 的索引来定位数据
                sample_data = data.loc[[idx]]
                true_label = labels.loc[idx]

                # 计算模型预测与真实标签的差异
                pred = model.predict(sample_data)
                model_change += abs(pred[0] - true_label)

            # 检查是否有更大的模型改变量
            if model_change > max_change:
                max_change = model_change
                selected_index = idx
                selected_model_change = model_change

        # 更新批次和未标记数据池
        if selected_index is not None:
            b_indices.append(selected_index)
            U_indices.remove(selected_index)
            print(f"Selected sample index {selected_index} with model change {selected_model_change}")

    return b_indices


def generate_super_features(data_point, models):
    """
    Generate super features for a given data point using trained GBDT models.

    Args:
        data_point (array-like): The data point for feature generation.
        models (list): List of trained GradientBoostingRegressor models.

    Returns:
        np.ndarray: Array containing the super features.
    """
    # Initialize an empty list to hold the outputs from each decision tree
    super_features = []

    # Loop through each model and collect its prediction as a super feature
    for model in models:
        # Here, we sum the contribution of each tree to get the final prediction
        # This could alternatively be done by calling model.predict() if you want the overall model output
        # but here we manually sum over each tree's contribution for illustrative purposes
        trees = model.estimators_.flatten()
        total_prediction = 0
        for tree in trees:
            leaf_index = tree.apply(data_point.reshape(1, -1))
            total_prediction += tree.tree_.value[leaf_index][0][0]
        super_features.append(total_prediction)

    return np.array(super_features)
