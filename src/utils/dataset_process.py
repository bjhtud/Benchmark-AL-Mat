#!/usr/bin/env python3
"""
Dataset Processing Utilities for Active Learning

This module contains functions for processing and preparing datasets
for active learning experiments in materials science. It provides
standardized data loading, cleaning, feature scaling, and train-validation
splitting functionality.

Author: [Your Name]
Email: [Your Email]
License: MIT License
Created: 2025-01-06
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_process_yin(file_path, random_state=36):
    """
    Process dataset for pullout/interface studies.

    This function loads data from a CSV file, drops missing values,
    removes specific target columns (Fmax, IFSS), standardizes features,
    and splits into training and validation sets with reset indices.

    Args:
        file_path (str): Path to the CSV data file.
        random_state (int, optional): Random seed for reproducible splits.
            Defaults to 36.

    Returns:
        tuple: (X_t, X_val, y_t, y_val) where:
            - X_t (pd.DataFrame): Standardized training features
            - X_val (pd.DataFrame): Standardized validation features
            - y_t (pd.Series): Training targets (Fmax values)
            - y_val (pd.Series): Validation targets (Fmax values)
    """
    # Load and clean data
    data = pd.read_csv(file_path)
    data = data.dropna()

    # Separate features and target
    X = data.drop(columns=['Fmax', 'IFSS'])
    y = data['Fmax']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Split data into training and validation sets
    X_t, X_val, y_t, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

    # Reset indices for consistency
    X_t = X_t.reset_index(drop=True)  # Drop old index
    y_t = y_t.reset_index(drop=True)  # Drop old index
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    return X_t, X_val, y_t, y_val


def data_process(file_path, target_columns, target_to_fit, random_state=36):
    """
    Generic data processing function for materials datasets.

    Loads data from CSV, cleans it, separates features and targets,
    standardizes features, and splits into training and validation sets.

    Args:
        file_path (str): Path to the CSV data file.
        target_columns (list): List of column names to be excluded from features.
        target_to_fit (str or list): Target column name(s) to use for prediction.
        random_state (int, optional): Random seed for reproducible splits.
            Defaults to 36.

    Returns:
        tuple: (X_t, X_val, y_t, y_val) where:
            - X_t (pd.DataFrame): Standardized training features
            - X_val (pd.DataFrame): Standardized validation features
            - y_t (pd.Series): Training targets
            - y_val (pd.Series): Validation targets
    """
    # Load and clean data
    data = pd.read_csv(file_path)
    data = data.dropna()

    # Separate features and target
    X = data.drop(columns=target_columns)
    y = data[target_to_fit]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Split data into training and validation sets
    X_t, X_val, y_t, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

    # Reset indices for consistency
    X_t = X_t.reset_index(drop=True)  # Drop old index
    y_t = y_t.reset_index(drop=True)  # Drop old index
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    return X_t, X_val, y_t, y_val


def data_process_meta(file_path, random_state=36, start_row=0):
    """
    Process multiple datasets based on metadata configuration.

    Reads dataset metadata from a CSV file and processes each dataset
    according to its configuration. Supports batch processing of multiple
    materials science datasets with different target configurations.

    Args:
        file_path (str): Path to the metadata CSV file containing dataset configurations.
        random_state (int, optional): Random seed for reproducible processing.
            Defaults to 36.
        start_row (int, optional): Starting row index for processing datasets.
            Defaults to 0.

    Returns:
        dict: Dictionary mapping dataset names to processed data tuples.
            Each value is [X_t, X_val, y_t, y_val] list containing:
            - X_t (pd.DataFrame): Standardized training features
            - X_val (pd.DataFrame): Standardized validation features
            - y_t (pd.Series): Training targets
            - y_val (pd.Series): Validation targets

    Note:
        The metadata CSV should contain columns: 'dataname', 'path',
        'target_columns', and 'target_to_fit'. Target columns should be
        semicolon-separated strings.
    """
    datasets = {}

    # Load and process metadata
    meta_data = pd.read_csv(file_path)
    meta_data = meta_data.iloc[start_row:]

    for index, row in meta_data.iterrows():
        try:
            # Parse configuration from metadata
            target_columns = row['target_columns'].split(';')
            target_to_fit = row['target_to_fit'].split(';')

            # Process dataset according to configuration
            X_t, X_val, y_t, y_val = data_process(
                row['path'], target_columns, target_to_fit, random_state
            )

            # Store processed dataset
            datasets[row['dataname']] = [X_t, X_val, y_t, y_val]

        except Exception as e:
            print(f"Warning: Failed to process dataset '{row['dataname']}': {str(e)}")
            continue

    return datasets
