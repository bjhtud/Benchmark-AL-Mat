#!/usr/bin/env python3
"""
Tree-Based Active Learning Strategy with Diversity Criterion

This module implements a wrapper for tree-based active learning using diversity
criteria in leaf nodes for sample selection.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
from .tree_diversity import Breiman_Tree_diversity



class TreeBasedRegressor_Diversity:
    """
    Tree-Based Regressor with Diversity-based Active Learning.

    This class implements a wrapper around Breiman_Tree_diversity for active
    learning in regression tasks using diversity-based sample selection.

    Attributes:
        random_state (int): Random seed for reproducibility.
        min_samples_leaf (int): Minimum samples required at leaf nodes.
    """

    def __init__(self, random_state=None, min_samples_leaf=5, **kwargs):
        """
        Initialize the TreeBasedRegressor_Diversity strategy.

        Args:
            random_state (int, optional): Random seed. Defaults to None.
            min_samples_leaf (int, optional): Minimum samples per leaf. Defaults to 5.
            **kwargs: Additional keyword arguments.
        """
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X_labeled, y_labeled):
        """
        Fit method for compatibility.

        Args:
            X_labeled: Labeled features (unused).
            y_labeled: Labeled targets (unused).
        """
        pass

    def query(self, X_labeled, X_unlabeled, y_labeled, y_unlabeled, n_act=1, **kwargs):
        """
        Query method for selecting samples using tree-based diversity strategy.

        Args:
            X_labeled (pd.DataFrame): Labeled data features.
            X_unlabeled (pd.DataFrame): Unlabeled data features.
            y_labeled (pd.DataFrame): Labeled data targets.
            y_unlabeled (pd.DataFrame): Unlabeled data targets.
            n_act (int, optional): Number of samples to select. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Returns:
            list: Indices of selected samples.
        """
        # 将 DataFrame 转为 numpy 数组
        X_labeled_np = X_labeled.to_numpy()
        X_unlabeled_np = X_unlabeled.to_numpy()
        y_labeled_np = y_labeled.to_numpy().flatten()

        # 保存未标注数据的索引
        unlabeled_index = X_unlabeled.index

        # 将已标注和未标注数据合并
        all_data = np.vstack((X_labeled_np, X_unlabeled_np))

        # 初始化 Breiman_Tree_diversity 类
        tree = Breiman_Tree_diversity(min_samples_leaf=self.min_samples_leaf, seed=self.random_state)

        # 输入数据
        labelled_indices = list(range(len(X_labeled_np)))  # 已标注数据的索引
        labels = list(y_labeled_np)  # 已标注样本的标签

        tree.input_data(all_data, labelled_indices, labels)

        # 训练树
        tree.fit_tree()

        # 计算叶子节点比例
        tree.al_calculate_leaf_proportions()

        # 选择新的点进行标注
        new_points = tree.pick_new_points(num_samples=n_act)

        # 从未标注数据中找到这些新点的绝对索引
        selected_indices = [unlabeled_index[i - len(X_labeled_np)] for i in new_points]

        # 确保selected_indices列表中没有numpy数据类型或者int64数据类型
        selected_indices = [int(i) for i in selected_indices]

        return selected_indices