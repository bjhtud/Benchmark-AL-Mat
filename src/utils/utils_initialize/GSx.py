import numpy as np


def greedy_initialize_x(df, n_samples):
    # 计算质心

    distances = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            distances[i] += distance(df.iloc[i].values, df.iloc[j].values)

    # 选择距离和最小的点作为第一个点
    centroid = np.argmin(distances)

    # 初始化已选择样本集合
    selected_samples_indices = []
    # 选择与质心最近的样本作为第一个样本
    first_sample_index = find_nearest_sample(df, centroid)
    selected_samples_indices.append(first_sample_index)
    remaining_samples = df.drop(first_sample_index)
    # 选择后续样本
    while len(selected_samples_indices) < n_samples:
        max_distance = -1
        next_sample_index = None
        for idx, sample in remaining_samples.iterrows():
            # 计算该样本到已选择样本集中每个样本的距离的最小值
            min_distance = min([distance(sample.values, df.loc[i].values) for i in selected_samples_indices])
            # 找出距离最远的那个样本
            if min_distance > max_distance:
                max_distance = min_distance
                next_sample_index = idx
        selected_samples_indices.append(next_sample_index)
        remaining_samples = remaining_samples.drop(next_sample_index)
        selected_samples_indices = [int(i) for i in selected_samples_indices]
    return selected_samples_indices


def find_nearest_sample(df, point):
    # 计算样本中距离point最近的样本
    nearest_sample_index = np.argmin(np.linalg.norm(df - point, axis=1))
    return df.index[nearest_sample_index]


def distance(sample1, sample2):
    # 计算样本之间的欧氏距离
    return np.linalg.norm(sample1 - sample2)
