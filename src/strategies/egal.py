#!/usr/bin/env python3
"""
EGAL (Expected Gradient-based Active Learning) Strategy

This module implements the EGAL active learning strategy for regression tasks,
which selects samples based on similarity and density measures.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys


class EGAL:
    """
    EGAL (Expected Gradient-based Active Learning) strategy.

    This class implements an active learning strategy that selects samples
    based on similarity and density calculations using cosine similarity.

    Attributes:
        labeled_indices (list): Indices of labeled samples.
        unlabeled_indices (list): Indices of unlabeled samples.
        beta (float): Threshold parameter for candidate selection.
        b_factor (float): Factor controlling candidate set size.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, b_factor=0.2, random_state=42):
        """
        Initialize the EGAL strategy.

        Args:
            b_factor (float, optional): Factor to control candidate set size. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducible results. Defaults to 42.
        """
        self.labeled_indices = None
        self.unlabeled_indices = None
        self.beta = None
        self.b_factor = b_factor
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit method for compatibility.

        Args:
            X: Input features (unused).
            y: Target values (unused).
        """
        pass

    def calculate_similarity_matrix(self, X_train_full_df):
        # Calculate cosine similarity for the full dataset
        return cosine_similarity(X_train_full_df)

    def calculate_alpha_beta(self, X_train_full_df):
        # Calculate alpha and set initial beta
        train_similarity_matrix = self.calculate_similarity_matrix(X_train_full_df)
        similarity_values = train_similarity_matrix[np.triu_indices_from(train_similarity_matrix, k=1)]
        mu = np.mean(similarity_values)
        delta = np.std(similarity_values)
        alpha = mu - 0.5 * delta
        self.beta = alpha
        return alpha

    def update_beta(self, unlabeled_to_labeled_similarity_matrix, b_factor=0.2, addendum_size=None):
        # Update beta to ensure the candidate set is large enough
        max_similarities = np.max(unlabeled_to_labeled_similarity_matrix, axis=1)
        sorted_similarities = np.sort(max_similarities)
        index = min(int(np.floor(b_factor * len(sorted_similarities))), len(sorted_similarities) - 1)
        beta = sorted_similarities[index]

        while len([sim for sim in max_similarities if sim <= beta]) < addendum_size:
            # Check if b_factor has reached its maximum value (1.0)
            if b_factor >= 1.0:
                print("Error: Unable to select enough candidates. Exiting program.")
                sys.exit(1)  # Exit the program with an error code

            # Increase b_factor and recalculate beta
            b_factor = min(1.0, b_factor + 0.05)
            index = min(int(np.floor(b_factor * len(sorted_similarities))), len(sorted_similarities) - 1)
            beta = sorted_similarities[index]

        # Update class beta to the final value
        self.beta = beta
        return self.beta

    def select_candidates(self, X_train_unlabeled_df, X_train_labeled_df, unlabeled_indices, addendum_size, b_factor):
        # Select candidate samples based on beta
        unlabeled_to_labeled_similarity_matrix = cosine_similarity(X_train_unlabeled_df, X_train_labeled_df)
        max_similarities = np.max(unlabeled_to_labeled_similarity_matrix, axis=1)
        candidate_indices = [idx for idx, similarity in zip(unlabeled_indices, max_similarities) if similarity <=
                             self.beta]

        while len(candidate_indices) < addendum_size:
            self.beta = self.update_beta(unlabeled_to_labeled_similarity_matrix, b_factor, addendum_size)
            candidate_indices = [idx for idx, similarity in zip(unlabeled_indices, max_similarities) if similarity <=
                                 self.beta]

        return candidate_indices

    def calculate_density(self, X_train_unlabeled_df, X_train_full_df, candidate_indices, alpha):
        # Calculate density for candidate samples
        candidate_density_scores = []
        for candidate_idx in candidate_indices:
            similarities = cosine_similarity([X_train_unlabeled_df.loc[candidate_idx]], X_train_full_df).flatten()
            neighborhood = similarities[similarities >= alpha]
            density = np.sum(neighborhood)
            candidate_density_scores.append(density)
        return candidate_density_scores

    def query(self, X_unlabeled, X_labeled, n_act=1, **kwargs):
        """
        Query method for selecting the next set of unlabeled samples.

        Args:
            X_unlabeled (pd.DataFrame): Features of unlabeled data.
            X_labeled (pd.DataFrame): Features of labeled data.
            n_act (int, optional): Number of samples to select. Defaults to 1.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list: Indices of the selected unlabeled samples.
        """

        X_train_full_df = pd.concat([X_labeled, X_unlabeled])
        self.unlabeled_indices = X_unlabeled.index.tolist()
        self.labeled_indices = X_labeled.index.tolist()

        # Step 1: Calculate alpha and beta
        alpha = self.calculate_alpha_beta(X_train_full_df)

        # Step 2: Select candidate samples based on beta
        candidate_indices = self.select_candidates(X_unlabeled, X_labeled, self.unlabeled_indices,
                                                   n_act, self.b_factor)

        # Step 3: Calculate density scores for candidates
        density_scores = self.calculate_density(X_unlabeled, X_train_full_df, candidate_indices, alpha)

        # Step 4: Select top samples by density score
        sorted_indices = np.argsort(density_scores)[-n_act:]  # Get the top addendum_size samples
        selected_indices = [int(candidate_indices[i]) for i in sorted_indices]

        return selected_indices
