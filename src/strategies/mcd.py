#!/usr/bin/env python3
"""
Monte Carlo Dropout (MCD) Active Learning Strategy

This module implements the Monte Carlo Dropout active learning strategy for regression
tasks, which uses dropout-enabled neural networks to estimate prediction uncertainty
and select the most uncertain samples for labeling.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from baal import ModelWrapper
from baal.bayesian import MCDropoutConnectModule
from sklearn.model_selection import train_test_split
from baal.modelwrapper import TrainingArgs

warnings.filterwarnings("ignore")


class mcdropout:
    """
    Monte Carlo Dropout (MCD) active learning strategy.

    This class implements the MCD approach which uses dropout during inference
    to estimate model uncertainty. Samples with the highest prediction uncertainty
    (standard deviation across multiple forward passes) are selected for labeling.

    Attributes:
        batch_size (int): Batch size for training and inference.
        num_epochs (int): Number of training epochs.
        iterations (int): Number of Monte Carlo iterations for uncertainty estimation.
        learning_rate (float): Learning rate for optimizer.
        dropout_rate (float): Dropout rate for uncertainty estimation.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, batch_size=8, num_epochs=100, iterations=100, learning_rate=0.001, dropout_rate=0.5,
                 random_state=42):
        """
        Initialize the Monte Carlo Dropout strategy.

        Args:
            batch_size (int, optional): Batch size for training. Defaults to 8.
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            iterations (int, optional): Number of MC iterations for uncertainty estimation.
                Defaults to 100.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
            dropout_rate (float, optional): Dropout rate for uncertainty estimation.
                Defaults to 0.5.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.random_state = random_state

    def fit(self, X_labeled, y_labeled):
        """
        Fit method for compatibility (training is done in query method).

        Args:
            X_labeled: Labeled feature data.
            y_labeled: Labeled target data.
        """
        pass  # Training is performed in the query method

    def query(self, X_unlabeled, X_labeled, y_unlabeled, y_labeled, n_act=1):
        """
        Query samples using Monte Carlo Dropout strategy.

        Trains a neural network with dropout and uses Monte Carlo sampling
        during inference to estimate prediction uncertainty. Selects samples
        with the highest uncertainty for active learning.

        Args:
            X_unlabeled (pd.DataFrame): Unlabeled feature data.
            X_labeled (pd.DataFrame): Labeled feature data.
            y_unlabeled (pd.Series): Unlabeled target data.
            y_labeled (pd.Series): Labeled target data.
            n_act (int, optional): Number of samples to query. Defaults to 1.

        Returns:
            list: Indices of selected samples with highest prediction uncertainty.
        """
        # Convert data to tensors
        X_training_tensor = torch.tensor(X_labeled.values, dtype=torch.float32)
        y_training_tensor = torch.tensor(y_labeled.values, dtype=torch.float32)
        X_pool = torch.tensor(X_unlabeled.values, dtype=torch.float32)

        # Split training data for validation
        X_train_mcd, X_val_mcd, y_train_mcd, y_val_mcd = train_test_split(
            X_training_tensor, y_training_tensor,
            test_size=0.2, random_state=self.random_state
        )

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_mcd, y_train_mcd)
        val_dataset = torch.utils.data.TensorDataset(X_val_mcd, y_val_mcd)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        # Define model architecture
        input_dim = X_training_tensor.shape[1]
        output_dim = y_training_tensor.shape[1]

        model = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim)
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        best_model_wts = None

        # Training loop with early stopping
        for epoch in range(self.num_epochs):
            model.train()  # Set model to training mode

            # Training phase
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()  # Switch model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for X_val_batch, y_val_batch in val_loader:
                    val_outputs = model(X_val_batch)
                    batch_loss = criterion(val_outputs, y_val_batch)
                    val_loss += batch_loss.item() * X_val_batch.size(0)

                val_loss /= len(val_loader.dataset)

            # Save best model weights based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = model.state_dict().copy()

        # Load best model weights
        model.load_state_dict(best_model_wts)

        # Add Monte Carlo Dropout for uncertainty estimation
        model = MCDropoutConnectModule(model, layers=['Linear'], weight_dropout=self.dropout_rate)
        wrapper = ModelWrapper(
            model=model,
            args=TrainingArgs(
                criterion=criterion,
                optimizer=optimizer,
                batch_size=self.batch_size,
                epoch=self.num_epochs,
                use_cuda=False
            )
        )

        # Perform Monte Carlo sampling for uncertainty estimation
        with torch.no_grad():
            predictions = wrapper.predict_on_batch(X_pool, iterations=self.iterations)

        # Calculate uncertainty (standard deviation across MC samples)
        reshaped_predictions = predictions.squeeze(1)
        std_per_row = torch.std(reshaped_predictions, dim=1)
        std_per_row_np = std_per_row.numpy()

        # Select samples with highest uncertainty
        uncertain_idx = np.argsort(std_per_row_np)[-n_act:].tolist()
        selected_indices = [int(i) for i in X_unlabeled.index[uncertain_idx]]
        return selected_indices
