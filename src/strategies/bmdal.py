#!/usr/bin/env python3
"""
Bayesian Model-Agnostic Deep Active Learning (BMDAL) Strategy

This module implements the BMDAL active learning strategy for regression tasks,
which uses Bayesian deep learning principles to select the most informative samples.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import os
import sys
import warnings

import torch
import torch.optim as optim
from torch import nn

from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch

warnings.filterwarnings("ignore")


class BMDAL:
    """
    Bayesian Model-Agnostic Deep Active Learning (BMDAL) strategy.

    This class implements the BMDAL approach which combines deep learning with
    Bayesian principles to select the most informative samples for active learning.
    Supports multiple selection methods including LCMD, MaxDiag, MaxDet, BAIT, etc.

    Attributes:
        random_state (int): Random seed for reproducibility.
        selection_method (str): Method for batch selection.
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of output targets.
        custom_model (nn.Module): Neural network model.
        criterion (nn.Module): Loss function for training.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.
        sigma (float): Noise parameter for kernel computations.
    """

    def __init__(self, random_state=42, selection_method='lcmd'):
        """
        Initialize the BMDAL strategy.

        Args:
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.
            selection_method (str, optional): Batch selection method. Options include
                'lcmd', 'maxdiag', 'maxdet', 'bait', 'maxdist', 'kmeanspp'.
                Defaults to 'lcmd'.
        """
        self.random_state = random_state
        self.input_dim = None
        self.output_dim = None
        self.custom_model = None
        self.selection_method = selection_method
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.num_epochs = 200
        self.sigma = 1e-3

    def fit(self, X_labeled, y_labeled):
        """
        Fit method for compatibility (training is done in query method).

        Args:
            X_labeled: Labeled feature data.
            y_labeled: Labeled target data.
        """
        pass  # Training is performed in the query method

    def query(self, X_unlabeled, X_labeled, y_labeled, y_unlabeled, n_act=1):
        """
        Query samples using BMDAL strategy.

        Trains a neural network on labeled data and uses Bayesian principles
        to select the most informative samples from the unlabeled pool.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled feature data.
            X_labeled (pd.DataFrame): Labeled feature data.
            y_labeled (pd.Series): Labeled target data.
            y_unlabeled: Unlabeled target data (unused).
            n_act (int, optional): Number of samples to query. Defaults to 1.

        Returns:
            list: Indices of selected samples based on BMDAL criteria.
        """
        # Convert data to tensors
        X_labeled_tensor = torch.tensor(X_labeled.values).float()
        y_labeled_tensor = torch.tensor(y_labeled.values).float()
        X_unlabeled_tensor = torch.tensor(X_unlabeled.values).float()

        # Initialize model dimensions
        self.input_dim = X_labeled.shape[1]
        self.output_dim = y_labeled.shape[1]

        # Create neural network model
        self.custom_model = nn.Sequential(
            nn.Linear(self.input_dim, 50), nn.ReLU(),
            nn.Linear(50, 25), nn.ReLU(),
            nn.Linear(25, self.output_dim)
        )
        self.optimizer = optim.Adam(self.custom_model.parameters(), lr=0.01)

        # Train the model
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self.custom_model(X_labeled_tensor)
            loss = self.criterion(outputs, y_labeled_tensor)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Prepare data for BMDAL selection
        train_data = TensorFeatureData(X_labeled_tensor)
        pool_data = TensorFeatureData(X_unlabeled_tensor)

        # Select batch based on chosen method
        if self.selection_method == 'lcmd':
            new_idxs, _ = select_batch(
                batch_size=n_act, models=[self.custom_model],
                data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                selection_method=self.selection_method, sel_with_train=True,
                base_kernel='grad', kernel_transforms=[('rp', [512])]
            )

        elif self.selection_method == 'maxdiag':
            new_idxs, _ = select_batch(
                batch_size=n_act, models=[self.custom_model],
                data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                selection_method=self.selection_method, sel_with_train=True,
                base_kernel='ll', kernel_transforms=[('train', [self.sigma, None])]
            )

        elif self.selection_method == 'maxdet':
            new_idxs, _ = select_batch(
                batch_size=n_act, models=[self.custom_model],
                data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                selection_method=self.selection_method, sel_with_train=True,
                base_kernel='ll', kernel_transforms=[('train', [self.sigma, None])]
            )

        elif self.selection_method == 'bait':
            new_idxs, _ = select_batch(
                batch_size=n_act, models=[self.custom_model],
                data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                selection_method=self.selection_method, sel_with_train=False,
                base_kernel='ll', kernel_transforms=[('train', [self.sigma, None])]
            )

        elif self.selection_method == 'maxdist':
            new_idxs, _ = select_batch(
                batch_size=n_act, models=[self.custom_model],
                data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                selection_method=self.selection_method, sel_with_train=True,
                base_kernel='ll', kernel_transforms=[]
            )

        elif self.selection_method == 'kmeanspp':
            new_idxs, _ = select_batch(batch_size=n_act, models=[self.custom_model],
                                       data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                                       selection_method=self.selection_method, sel_with_train=False,
                                       base_kernel='ll', kernel_transforms=[('train', [self.sigma, None])])

        elif self.selection_method == 'fw':
            new_idxs, _ = select_batch(batch_size=n_act, models=[self.custom_model],
                                       data={'train': train_data, 'pool': pool_data}, y_train=y_labeled_tensor,
                                       selection_method=self.selection_method, sel_with_train=False,
                                       base_kernel='ll', kernel_transforms=[('acs-rf-hyper', [512, None])])

        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        # Convert to original indices
        selected_indices = [int(X_unlabeled.index[idx]) for idx in new_idxs]
        return selected_indices
