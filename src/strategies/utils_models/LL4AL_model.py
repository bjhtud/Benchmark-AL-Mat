#!/usr/bin/env python3
"""
LL4AL Neural Network Models

This module implements the neural network architectures for the LL4AL
(Learning Loss for Active Learning) strategy, including the main network
and the loss prediction network.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MainNet(nn.Module):
    """
    Main neural network for regression tasks.

    This network serves as the primary model for making predictions
    and extracting intermediate features for loss prediction.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer.
    """

    def __init__(self, input_dim):
        """
        Initialize the MainNet.

        Args:
            input_dim (int): Input feature dimension.
        """
        super(MainNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Prediction and intermediate features [x1, x2].
        """
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x = self.fc3(x2)
        return x, [x1, x2]  # Return prediction result and intermediate features


class LossNet(nn.Module):
    """
    Loss prediction network for LL4AL strategy.

    This network predicts the loss values based on intermediate features
    from the main network to guide active learning sample selection.

    Attributes:
        fc1 (nn.Linear): First feature transformation layer.
        fc2 (nn.Linear): Second feature transformation layer.
        linear (nn.Linear): Final output layer.
    """

    def __init__(self, interm_dim=16):
        """
        Initialize the LossNet.

        Args:
            interm_dim (int, optional): Intermediate dimension. Defaults to 16.
        """
        super(LossNet, self).__init__()
        # Define fully connected layers for each intermediate feature
        self.fc1 = nn.Linear(50, interm_dim)
        self.fc2 = nn.Linear(25, interm_dim)
        # self.fc3 = nn.Linear(16, interm_dim)

        # Final fully connected layer that maps concatenated features to scalar value
        self.linear = nn.Linear(2 * interm_dim, 1)

    def forward(self, features):
        """
        Forward pass through the loss prediction network.

        Args:
            features (list): List of intermediate features from MainNet.

        Returns:
            torch.Tensor: Predicted loss value.
        """
        # Extract each intermediate feature and pass through fully connected layers
        x1 = F.relu(self.fc1(features[0]))
        x2 = F.relu(self.fc2(features[1]))
        # x3 = F.relu(self.fc3(features[2]))

        # Concatenate all features
        # x = torch.cat((x1, x2, x3), dim=1)
        x = torch.cat((x1, x2), dim=1)
        # Pass through final fully connected layer to get loss prediction
        x = self.linear(x)
        return x