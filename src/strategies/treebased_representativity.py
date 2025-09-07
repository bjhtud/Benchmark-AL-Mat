#!/usr/bin/env python3
"""
Tree-Based Active Learning Strategy with Representativity Criterion

This module implements a wrapper for tree-based active learning using representativity
criteria in leaf nodes for sample selection.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
from .tree_representativity import Breiman_Tree_representativity


class TreeBasedRegressor_Representativity:
    """
    Tree-Based Regressor with Representativity-based Active Learning.

    This class implements a wrapper around Breiman_Tree_representativity for active
    learning in regression tasks using representativity-based sample selection.

    Attributes:
        random_state (int): Random seed for reproducibility.
        min_samples_leaf (int): Minimum samples required at leaf nodes.
    """

    def __init__(self, random_state=None, min_samples_leaf=5, **kwargs):
        """
        Initialize the TreeBasedRegressor_Representativity strategy.

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
        Query method for selecting samples using tree-based representativity strategy.

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

        # Convert DataFrame to numpy arrays
        X_labeled_np = X_labeled.to_numpy()
        X_unlabeled_np = X_unlabeled.to_numpy()
        y_labeled_np = y_labeled.to_numpy().flatten()

        # Save indices of unlabeled data
        unlabeled_index = X_unlabeled.index

        # Combine labeled and unlabeled data
        all_data = np.vstack((X_labeled_np, X_unlabeled_np))

        # Initialize Breiman_Tree_representativity class
        tree = Breiman_Tree_representativity(min_samples_leaf=self.min_samples_leaf, seed=self.random_state)

        # Input data
        labelled_indices = list(range(len(X_labeled_np)))  # Indices of labeled data
        labels = list(y_labeled_np)  # Labels of labeled samples

        tree.input_data(all_data, labelled_indices, labels)

        # Train tree
        tree.fit_tree()

        # Calculate leaf node proportions
        tree.al_calculate_leaf_proportions()

        # Select new points for labeling
        new_points = tree.pick_new_points(num_samples=n_act)

        # Find absolute indices of these new points from unlabeled data
        selected_indices = [unlabeled_index[i - len(X_labeled_np)] for i in new_points]

        # Ensure selected_indices list contains no numpy data types or int64 data types
        selected_indices = [int(i) for i in selected_indices]

        return selected_indices