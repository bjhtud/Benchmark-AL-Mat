from .utils_initialize import greedy_initialize_x, random_initialize, kmeans_initialize, ncc_initialize


def initialize(X_unlabeled, n_samples, method='random', random_state=42):
    """
    初始化未标记样本集。
    :param X_unlabeled:     未标记样本集
    :param n_samples:       样本数
    :param method:          初始化方法
    :param random_state:    随机种子
    :return:                未标记样本集的索引
    """
    if method == 'random':
        return random_initialize(X_unlabeled, n_samples, random_state)
    elif method == 'greedy_search':
        return greedy_initialize_x(X_unlabeled, n_samples)
    elif method == 'kmeans':
        return kmeans_initialize(X_unlabeled, n_samples, random_state)
    elif method == 'ncc':
        return ncc_initialize(X_unlabeled, n_samples)
    else:
        raise ValueError("Invalid initialization method.")
