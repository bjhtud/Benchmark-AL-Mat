import numpy as np
from sklearn.cluster import KMeans


def kmeans_initialize(X_unlabeled, n_samples, random_state=42):
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state)
    kmeans.fit(X_unlabeled)
    # 从每个聚类中选择最接近其中心点的样本，并返回其索引
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    selected_indices = []
    for i in range(n_samples):
        # 获取属于该聚类的所有样本的位置索引
        cluster_indices = np.where(labels == i)[0]

        # 用位置索引来获取实际的DataFrame索引
        actual_indices = X_unlabeled.index[cluster_indices]
        # 计算这些样本与聚类中心的距离
        distances = np.linalg.norm(X_unlabeled.iloc[cluster_indices] - centers[i], axis=1)
        # 找到距离最小的样本的位置索引
        closest_index = np.argmin(distances)
        # 从实际索引中选择对应的索引
        selected_index = actual_indices[closest_index]
        selected_indices.append(selected_index)
        selected_indices = [int(i) for i in selected_indices]
    return selected_indices