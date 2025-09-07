import numpy as np


def ncc_initialize(X_unlabeled, n_samples):
    # 初始化已选择样本集合
    selected_samples_indices = []
    remaining_samples = X_unlabeled.copy()

    while len(selected_samples_indices) < n_samples:
        print(f'已选择样本数：{len(selected_samples_indices)}')
        min_total_distance = float('inf')
        next_sample_index = None

        for idx, sample in remaining_samples.iterrows():
            # 假设当前样本为下一选择样本
            temp_selected_indices = selected_samples_indices + [idx]
            total_distance = compute_total_nearest_neighbor_distance(X_unlabeled, temp_selected_indices)

            # 找出最近邻距离总和最小的那个样本
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                next_sample_index = idx

        selected_samples_indices.append(next_sample_index)
        remaining_samples = remaining_samples.drop(next_sample_index)
        selected_samples_indices = [int(i) for i in selected_samples_indices]
    return selected_samples_indices


def compute_total_nearest_neighbor_distance(df, selected_indices):
    total_distance = 0
    for idx, sample in df.iterrows():
        if idx not in selected_indices:
            min_distance = min([distance(sample.values, df.loc[i].values) for i in selected_indices])
            total_distance += min_distance
    return total_distance


def distance(sample1, sample2):
    # 计算样本之间的欧氏距离
    return np.linalg.norm(sample1 - sample2)