import numpy as np


def random_initialize(X_unlabeled, n_samples, random_state):
    rng = np.random.default_rng(seed=random_state)
    initial_idx = rng.choice(range(len(X_unlabeled)), size=n_samples, replace=False)
    initial_idx = X_unlabeled.index[initial_idx]
    initial_idx = [int(i) for i in initial_idx]
    return initial_idx