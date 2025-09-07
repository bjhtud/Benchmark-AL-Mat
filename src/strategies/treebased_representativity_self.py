#!/usr/bin/env python3
"""
Tree-Based Active Learning Strategy with Representativity Criterion (Self-implemented)

This module implements a self-contained tree-based active learning strategy
with representativity-based sample selection for regression tasks.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor


class TreeBasedRegressor_Representativity_self(DecisionTreeRegressor):
    """
    Tree-Based Regressor with Representativity-based Active Learning (Self-implemented).

    This class implements a self-contained active learning strategy that uses
    decision trees with representativity-based sample selection in leaf nodes.

    Attributes:
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state=None, **kwargs):
        """
        Initialize the TreeBasedRegressor_Representativity_self strategy.

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

        unique_leaves, counts = np.unique(leaf_indices, return_counts=True)
        for leaf, count in zip(unique_leaves, counts):
            pi_k = count / N
            leaf_pis[leaf] = pi_k

        return leaf_pis

    def calculate_leaf_variances(self, X_labeled, y_labeled):
        """
        Calculate the variance of each leaf node.

        Args:
            X_labeled (pd.DataFrame): Labeled sample set (DataFrame).
            y_labeled (pd.Series or pd.DataFrame): True labels of labeled samples (Series or DataFrame).

        Returns:
            dict: Variance of each leaf node.
        """
        # Get leaf node indices
        leaf_indices = self.apply(X_labeled)
        if leaf_indices.ndim > 1:
            leaf_indices = leaf_indices.ravel()  # Flatten to 1D array

        # Predict values for labeled samples
        y_pred = self.predict(X_labeled)

        # Calculate residuals
        if isinstance(y_labeled, pd.DataFrame):
            # If y_labeled is DataFrame, use .squeeze() to convert it to Series
            y_labeled_series = y_labeled.squeeze()
        else:
            y_labeled_series = y_labeled

        residuals = y_labeled_series.values - y_pred
        if residuals.ndim > 1:
            residuals = residuals.ravel()  # Flatten to 1D array

        # Create DataFrame
        df = pd.DataFrame({'leaf': leaf_indices, 'residual': residuals})

        # Calculate variance for each leaf node
        leaf_variances = {}
        for leaf, group in df.groupby('leaf'):
            if len(group) > 1:
                variance = group['residual'].var(ddof=1)
            else:
                variance = 1e-6  # Avoid zero variance
            leaf_variances[leaf] = variance

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
        sqrt_terms = [np.sqrt(pi * sigma2) for pi, sigma2 in zip(probabilities.values(), variances.values())]
        denominator = sum(sqrt_terms)

        if denominator == 0:
            # If denominator is zero, distribute samples evenly
            n_leaves = len(probabilities)
            samples_per_leaf = {leaf: n_act / n_leaves for leaf in probabilities.keys()}
        else:
            samples_per_leaf = {
                leaf: n_act * (np.sqrt(probabilities[leaf] * variances[leaf]) / denominator)
                for leaf in probabilities.keys()
            }

        # Round sample counts and adjust sum
        rounded_samples = {leaf: int(round(n)) for leaf, n in samples_per_leaf.items()}
        total_samples = sum(rounded_samples.values())
        difference = n_act - total_samples

        # Adjust sample counts to match n_act
        while difference != 0:
            for leaf in rounded_samples.keys():
                if difference == 0:
                    break
                if difference > 0:
                    rounded_samples[leaf] += 1
                    difference -= 1
                elif difference < 0 and rounded_samples[leaf] > 0:
                    rounded_samples[leaf] -= 1
                    difference += 1

        return rounded_samples

    def select_samples_to_label(self, samples_per_leaf, leaf_indices_unlabeled, X_unlabeled, X_labeled):
        """
        Select samples to label based on representativity and diversity sampling strategy.

        Args:
            samples_per_leaf (dict): Number of samples each leaf node should label.
            leaf_indices_unlabeled (array): Leaf node indices corresponding to unlabeled samples.
            X_unlabeled (pd.DataFrame): Unlabeled sample features (DataFrame).
            X_labeled (pd.DataFrame): Labeled sample features (DataFrame).

        Returns:
            list: List of selected sample indices for labeling.
        """
        selected_indices = []  # Store indices of samples selected for labeling

        # Get indices of labeled samples
        labeled_indices = X_labeled.index.tolist()

        # Initialize set of selected representative sample indices
        representative_indices = []

        for leaf_id, n_to_label in samples_per_leaf.items():
            if n_to_label <= 0:
                continue

            # Get unlabeled sample indices in current leaf node
            current_leaf_mask = (leaf_indices_unlabeled == leaf_id)
            current_leaf_indices = X_unlabeled.index[current_leaf_mask]

            # If number of unlabeled samples is less than needed, select all
            if len(current_leaf_indices) <= n_to_label:
                selected_indices.extend(current_leaf_indices.tolist())
                representative_indices.extend(current_leaf_indices.tolist())
                continue

            # Cluster unlabeled samples
            n_clusters = min(n_to_label, len(current_leaf_indices))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            kmeans.fit(X_unlabeled.loc[current_leaf_indices])

            # Initialize representative sample indices for each cluster
            cluster_labels = kmeans.labels_
            clusters = {}
            for idx, label in zip(current_leaf_indices, cluster_labels):
                clusters.setdefault(label, []).append(idx)

            # Initialize selected sample set, including labeled samples and representative samples
            selected_set = set(labeled_indices + representative_indices)

            # Select representative sample for each cluster
            for cluster_indices in clusters.values():
                best_score = -np.inf
                best_idx = None

                for idx in cluster_indices:
                    x_j = X_unlabeled.loc[idx].values.reshape(1, -1)

                    # Calculate representativity R(x_j)
                    other_indices = [i for i in cluster_indices if i != idx]
                    if other_indices:
                        distances_same_cluster = np.linalg.norm(
                            X_unlabeled.loc[other_indices].values - x_j, axis=1)
                        R_xj = distances_same_cluster.mean()
                    else:
                        R_xj = 0

                    # Calculate diversity Δ(x_j)
                    if selected_set:
                        # Get selected samples from X_unlabeled and X_labeled
                        selected_unlabeled_indices = list(selected_set.intersection(X_unlabeled.index))
                        selected_samples = pd.concat([
                            X_labeled,
                            X_unlabeled.loc[selected_unlabeled_indices]
                        ])
                        distances_selected = np.linalg.norm(
                            selected_samples.values - x_j, axis=1)
                        Delta_xj = distances_selected.min()
                    else:
                        Delta_xj = 0

                    score = Delta_xj - R_xj

                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    representative_indices.append(best_idx)
                    selected_set.add(best_idx)

            # If number of samples to select exceeds number of clusters, select additional samples
            remaining_to_select = n_to_label - len(clusters)
            if remaining_to_select > 0:
                # Select from remaining unselected samples
                remaining_indices = set(current_leaf_indices) - set(representative_indices)
                for _ in range(remaining_to_select):
                    best_score = -np.inf
                    best_idx = None

                    for idx in remaining_indices:
                        x_j = X_unlabeled.loc[idx].values.reshape(1, -1)

                        # Calculate representativity R(x_j)
                        distances_same_leaf = np.linalg.norm(
                            X_unlabeled.loc[current_leaf_indices].values - x_j, axis=1)
                        R_xj = distances_same_leaf.mean()

                        # Calculate diversity Δ(x_j)
                        if selected_set:
                            selected_unlabeled_indices = list(selected_set.intersection(X_unlabeled.index))
                            selected_samples = pd.concat([
                                X_labeled,
                                X_unlabeled.loc[selected_unlabeled_indices]
                            ])
                            distances_selected = np.linalg.norm(
                                selected_samples.values - x_j, axis=1)
                            Delta_xj = distances_selected.min()
                        else:
                            Delta_xj = 0

                        score = Delta_xj - R_xj

                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx is not None:
                        selected_indices.append(best_idx)
                        representative_indices.append(best_idx)
                        selected_set.add(best_idx)
                        remaining_indices.remove(best_idx)
                    else:
                        break  # Cannot select more samples

        return selected_indices

    def query(self, X_unlabeled, n_act=1, X_labeled=None, y_labeled=None, **kwargs):
        """
        Query samples using tree-based representativity strategy.

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

        # Determine the number of samples to select from each leaf node
        samples_per_leaf = self.calculate_samples_per_leaf(probabilities, variances, n_act)

        # Get leaf node indices of unlabeled samples
        leaf_indices_unlabeled = self.apply(X_unlabeled)

        # Select samples for labeling
        selected_indices = self.select_samples_to_label(samples_per_leaf, leaf_indices_unlabeled, X_unlabeled, X_labeled)

        return selected_indices
