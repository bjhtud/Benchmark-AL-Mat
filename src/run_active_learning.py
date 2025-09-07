#!/usr/bin/env python3
"""
Active Learning Benchmark for Materials Science

This script runs active learning benchmarks on various materials datasets
using different query strategies for regression tasks.

Author: [Jinghou Bi]
Email: [jinghou.bi@tu-dresden.de]
License: [MIT License]
"""

import argparse
import json
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from tqdm import tqdm

from utils import data_process_meta, active_learning
from strategies import (
    TreeBasedRegressor_Diversity, TreeBasedRegressor_Representativity, GaussianProcessBased,
    QueryByCommittee, Basic_RD_ALR, GSBAG, QDD, RandomSearch, GSi, LearningLoss, BMDAL,
    RD_GS_ALR, RD_QBC_ALR, RD_EMCM_ALR, EGAL, GSx, GSy, mcdropout,
    TreeBasedRegressor_Diversity_self, TreeBasedRegressor_Representativity_self
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('active_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


class ActiveLearningBenchmark:
    """Main class for running active learning benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark with configuration.

        Args:
            config: Dictionary containing benchmark configuration
        """
        self.config = config
        self.random_state = config['random_state']
        self.initial_method = config['initial_method']
        # self.strategy_idx = config['strategy_idx']
        self.strategy_name = config['strategy']
        # self.dataset_idx = config['dataset_idx']
        self.dataset_name = config['dataset']
        self.n_pro_query = config['n_pro_query']

        # Set up paths
        self.result_path = self._setup_result_paths()

        # Initialize kernel and other parameters
        self.kernel = self._create_kernel()
        self.threshold = config.get('threshold', 0.85)

        # Set random seed for reproducibility
        self._set_seed(self.random_state)

        logger.info(f"Initialized benchmark with config: {config}")

    def _setup_result_paths(self) -> str:
        """Create and return result directory paths."""
        result_path = os.path.join(
            "..", "result",
            str(self.n_pro_query),
            str(self.random_state),
            self.initial_method
        )

        time_record_path = os.path.join(result_path, 'time_record')

        for path in [result_path, time_record_path]:
            os.makedirs(path, exist_ok=True)

        logger.info(f"Result path: {result_path}")
        logger.info(f"Time record path: {time_record_path}")

        return result_path

    def _create_kernel(self) -> object:
        """Create and return the Gaussian process kernel."""
        return (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) +
                WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {seed}")

    def _create_estimators(self) -> List[object]:
        """Create and return list of active learning estimators."""
        estimators = [
            RandomSearch(random_state=self.random_state),
            GSBAG(random_state=self.random_state, kernel=self.kernel, n_restarts_optimizer=10),
            QueryByCommittee(random_state=self.random_state, num_learner=100),
            TreeBasedRegressor_Diversity(random_state=self.random_state, min_samples_leaf=5),
            TreeBasedRegressor_Representativity(random_state=self.random_state, min_samples_leaf=5),
            TreeBasedRegressor_Diversity_self(random_state=self.random_state, min_samples_leaf=5),
            TreeBasedRegressor_Representativity_self(random_state=self.random_state, min_samples_leaf=5),
            GaussianProcessBased(random_state=self.random_state, kernel=self.kernel, n_restarts_optimizer=10),
            QDD(random_state=self.random_state),
            GSi(random_state=self.random_state),
            GSx(random_state=self.random_state),
            GSy(random_state=self.random_state),
            LearningLoss(
                BATCH=16, LR=0.01, MARGIN=1, WEIGHT=0.0001,
                EPOCH=200, EPOCHL=75, WDECAY=5e-4,
                random_state=self.random_state
            ),
            EGAL(b_factor=0.25, random_state=self.random_state),
            mcdropout(
                random_state=self.random_state, learning_rate=0.01,
                num_epochs=200, batch_size=16
            ),
            Basic_RD_ALR(random_state=self.random_state),
            RD_GS_ALR(random_state=self.random_state),
            RD_QBC_ALR(random_state=self.random_state, num_learner=100),
            RD_EMCM_ALR(random_state=self.random_state),
            BMDAL(random_state=self.random_state, selection_method='lcmd'),
        ]
        return estimators

    def load_datasets(self) -> Dict[str, Tuple]:
        """Load and return datasets."""
        try:
            datasets = data_process_meta('../dataset/meta.csv', random_state=self.random_state)
            logger.info(f"Loaded {len(datasets)} datasets")
            return datasets
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

    def run_benchmark(self) -> None:
        """Run the active learning benchmark."""
        try:
            # Load datasets and estimators
            datasets = self.load_datasets()
            estimators = self._create_estimators()

            # Get specific dataset and estimator
            datasets_list = list(datasets.items())
            if self.dataset_idx >= len(datasets_list):
                raise ValueError(f"Dataset index {self.dataset_idx} out of range (0-{len(datasets_list)-1})")

            if self.strategy_idx >= len(estimators):
                raise ValueError(f"Strategy index {self.strategy_idx} out of range (0-{len(estimators)-1})")

            dataset_name, dataset_value = datasets_list[self.dataset_idx]
            selected_estimator = estimators[self.strategy_idx]

            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"Using strategy: {selected_estimator.__class__.__name__}")

            # Prepare data
            X_t, X_val, y_t, y_val = dataset_value

            # Check for duplicates
            duplicate_samples = X_t.duplicated()
            if duplicate_samples.any():
                logger.warning(f"Found {duplicate_samples.sum()} duplicate samples")

            # Calculate experiment parameters
            total_data_volume = X_t.shape[0]
            n_initial = 10
            n_queries = max(1, (total_data_volume - n_initial - 1) // self.n_pro_query)

            logger.info(f"Total data volume: {total_data_volume}")
            logger.info(f"Initial samples: {n_initial}")
            logger.info(f"Queries per iteration: {self.n_pro_query}")
            logger.info(f"Number of query iterations: {n_queries}")

            # Run active learning
            test_methods = ["normal"]  # Can be made configurable

            query_idx_all, query_time_all = active_learning(
                [selected_estimator], X_t, y_t, X_val, y_val,
                n_initial, self.n_pro_query, n_queries, self.threshold,
                initial_method=self.initial_method,
                test_methods=test_methods,
                random_state=self.random_state
            )

            # Save results
            self._save_results(
                query_idx_all, query_time_all,
                dataset_name, selected_estimator
            )

            logger.info("Benchmark completed successfully")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def _save_results(self, query_idx_all: Dict, query_time_all: Dict,
                     dataset_name: str, estimator: object) -> None:
        """Save benchmark results to JSON files."""
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Prepare result data
        result = query_idx_all.copy()
        result.update({
            'dataset_name': dataset_name,
            'initial_method': self.initial_method,
            'random_state': self.random_state,
            'strategy_name': estimator.__class__.__name__,
            'timestamp': current_time
        })

        result_time_record = query_time_all.copy()
        result_time_record.update({
            'dataset_name': dataset_name,
            'initial_method': self.initial_method,
            'random_state': self.random_state,
            'strategy_name': estimator.__class__.__name__,
            'timestamp': current_time
        })

        # Save main results
        result_filename = os.path.join(
            self.result_path,
            f"{estimator.__class__.__name__}_{dataset_name}_{current_time}.json"
        )

        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {result_filename}")

        # Save time records
        time_record_path = os.path.join(self.result_path, 'time_record')
        time_filename = os.path.join(
            time_record_path,
            f"{estimator.__class__.__name__}_{dataset_name}_{current_time}.json"
        )

        with open(time_filename, 'w', encoding='utf-8') as f:
            json.dump(result_time_record, f, indent=2, ensure_ascii=False)

        logger.info(f"Time records saved to {time_filename}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run active learning benchmark on materials datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--random-state", type=int, required=True,
        help="Random state for reproducibility"
    )

    parser.add_argument(
        "--initial-method", type=str, default='random',
        choices=['random', 'greedy_search', 'kmeans', 'ncc'],
        help="Initial sampling method for active learning"
    )

    # parser.add_argument(
    #     "--strategy-idx", type=int, required=True,
    #     help="Index of the active learning strategy to use (0-based)"
    # )
    #
    # parser.add_argument(
    #     "--dataset-idx", type=int, required=True,
    #     help="Index of the dataset to use (0-based)"
    # )

    # NEW: names instead of indices
    parser.add_argument(
        "--strategy", type=str, required=True,
        help="Name of the active learning strategy (case-insensitive, e.g., 'RD_EMCM_ALR')"
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Name of the dataset (case-insensitive, e.g., 'uci-concrete')"
    )

    parser.add_argument(
        "--n-pro-query", type=int, default=10,
        help="Number of samples to query per iteration"
    )

    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Performance threshold for early stopping"
    )

    parser.add_argument(
        "--config-file", type=str,
        help="Path to JSON configuration file (overrides command line args)"
    )

    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise

def main():
    """Main entry point."""
    args = parse_arguments()

    # Load configuration
    if args.config_file:
        config = load_config_file(args.config_file)
    else:
        config = {
            'random_state': args.random_state,
            'initial_method': args.initial_method,
            'strategy_idx': args.strategy_idx,
            'dataset_idx': args.dataset_idx,
            'n_pro_query': args.n_pro_query,
            'threshold': args.threshold
        }

    # Run benchmark
    benchmark = ActiveLearningBenchmark(config)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()

