#!/usr/bin/env python3
"""
Tree-Based Active Learning Strategy with Diversity Criterion (Self-implemented)

This module implements a self-contained tree-based active learning strategy
with diversity-based sample selection for regression tasks.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class TreeBasedRegressor_Diversity_self(DecisionTreeRegressor):
    """
    Tree-Based Regressor with Diversity-based Active Learning (Self-implemented).

    This class implements a self-contained active learning strategy that uses
    decision trees with diversity-based sample selection in leaf nodes.

    Attributes:
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state=None, **kwargs):
        """
        Initialize the TreeBasedRegressor_Diversity_self strategy.

        Args:
            random_state (int, optional): Random seed. Defaults to None.
            **kwargs: Additional keyword arguments for DecisionTreeRegressor.
        """
        super().__init__(random_state=random_state, **kwargs)
        self.random_state = random_state

    def calculate_leaf_probabilities(self, X_unlabeled):
        """
        Calculate the probability of unlabeled samples belonging to each leaf node.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled sample set.

        Returns:
            dict: Probability of each leaf node.
        """
        leaf_indices = self.apply(X_unlabeled)
        N = len(X_unlabeled)
        leaf_pis = {}

        for leaf in np.unique(leaf_indices):
            # Get sample indices that are not in the initial index set I_init and are in the current leaf region
            indices = [i for i in range(N) if leaf_indices[i] == leaf]

            # Calculate π_k value
            pi_k = len(indices) / N
            leaf_pis[leaf] = pi_k

        return leaf_pis

    def calculate_leaf_variances(self, X_labeled, y_labeled):
        """
        Calculate the variance of each leaf node.

        Args:
            X_labeled (pd.DataFrame): Labeled sample set.
            y_labeled (pd.Series): True labels of labeled samples.

        Returns:
            dict: Variance of each leaf node.
        """
        leaf_samples = {}
        y_training_predicted = self.predict(X_labeled)
        leaf_indices = self.apply(X_labeled)
        leaf_variances = {}

        for leaf_index in np.unique(leaf_indices):
            samples_in_leaf = np.where(leaf_indices == leaf_index)[0]
            targets_in_leaf = y_labeled.iloc[samples_in_leaf].values
            predict_in_leaf = y_training_predicted[samples_in_leaf]
            leaf_variances[leaf_index] = np.var(targets_in_leaf - predict_in_leaf, ddof=1)

        return leaf_variances

    def calculate_samples_per_leaf(self, probabilities, variances, n_act):
        """
        Calculate the number of samples each leaf should select.

        Args:
            probabilities (dict): Dictionary of sample proportions for each leaf node, with leaf index as key.
            variances (dict): Dictionary of estimated variance for each leaf node, with leaf index as key.
            n_act (int): Total number of samples to select.

        Returns:
            dict: Dictionary containing the number of samples each leaf should select.
        """
        # Calculate denominator part
        denominator = sum(np.sqrt(pi * sigma2) for pi, sigma2 in zip(probabilities.values(), variances.values()))

        # Calculate sample count for each leaf
        samples_per_leaf = {leaf: n_act * (np.sqrt(pi * sigma2) / denominator)
                            for leaf, (pi, sigma2) in
                            zip(probabilities.keys(), zip(probabilities.values(), variances.values()))}

        rounded_samples = {k: round(v) for k, v in samples_per_leaf.items()}
        rounded_sum = sum(rounded_samples.values())

        # Calculate difference
        difference = n_act - rounded_sum

        # If difference is not zero, need adjustment
        if difference != 0:
            # Sort dictionary items by the difference between before and after rounding
            sorted_samples = sorted(samples_per_leaf.items(), key=lambda x: x[1] - round(x[1]))

            # Adjust for the absolute value of difference times
            for i in range(abs(difference)):
                k, v = sorted_samples[i]
                if difference > 0:
                    rounded_samples[k] += 1
                else:
                    rounded_samples[k] -= 1

        return rounded_samples

    def select_samples_to_label(self, n_samples_to_label_per_leaf, leaf_indices, X_labeled, X_unlabeled):
        """
        Select samples for labeling in each leaf node.

        Args:
            n_samples_to_label_per_leaf (dict): Number of samples each leaf node should label.
            leaf_indices (array): Leaf node indices corresponding to unlabeled samples.
            X_labeled (pd.DataFrame): Labeled sample features.
            X_unlabeled (pd.DataFrame): Unlabeled sample features.

        Returns:
            list: Selected sample indices for labeling.
        """
        selected_indices = []  # Store indices of samples selected for labeling

        for leaf_id, n_to_label in n_samples_to_label_per_leaf.items():
            # Get indices of unlabeled samples in the current leaf node
            current_leaf_indices = np.where(leaf_indices == leaf_id)[0]

            # Get unlabeled sample features in the current leaf node
            current_leaf_unlabeled = X_unlabeled.iloc[current_leaf_indices]
            # Calculate distances between unlabeled samples and labeled samples
            distances = np.sqrt(
                ((current_leaf_unlabeled.values[:, np.newaxis, :] - X_labeled.values) ** 2).sum(axis=2))
            # Get minimum distances
            min_distances = np.min(distances, axis=1)
            # Select the n_to_label samples with largest distances
            farthest_indices = np.argsort(-min_distances)[:n_to_label]
            # Add to the final selected index list
            selected_indices.extend(current_leaf_indices[farthest_indices])

        return selected_indices

    def query(self, X_unlabeled, n_act=1, X_labeled=None, y_labeled=None, **kwargs):
        """
        Query samples using tree-based diversity strategy.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled data features.
            n_act (int, optional): Number of samples to select. Defaults to 1.
            X_labeled (pd.DataFrame): Labeled data features.
            y_labeled (pd.Series): Labeled data targets.
            **kwargs: Additional keyword arguments.

        Returns:
            list: Indices of selected samples.
        """
        self.fit(X_labeled, y_labeled)
        # Calculate leaf node probabilities and variances
        probabilities = self.calculate_leaf_probabilities(X_unlabeled)
        variances = self.calculate_leaf_variances(X_labeled, y_labeled)

        # Determine the number of new samples to label in active learning
        samples_to_label_per_leaf = self.calculate_samples_per_leaf(probabilities, variances, n_act)

        # Get leaf node indices of samples
        leaf_indices = self.apply(X_unlabeled)

        # Select samples for labeling
        selected_indices = self.select_samples_to_label(samples_to_label_per_leaf, leaf_indices, X_labeled, X_unlabeled)

        # Convert position indices to data indices
        selected_indices = X_unlabeled.index[selected_indices]

        selected_indices = [int(i) for i in selected_indices]
        return selected_indices
