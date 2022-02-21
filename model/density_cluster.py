import numpy as np

from model.argrelextrema_pro import argrelmax_pro, argrelmin_pro


def cluster_based_density(array, max_normal_deviation):
    # 获取直方图密度和边界
    hists, bins = np.histogram(array, bins="auto")

    # 获取直方图密度极大值和极小值的索引
    extreme_max_idx = argrelmax_pro(hists)
    extreme_min_idx = argrelmin_pro(hists)

    # 获取簇心和边界
    centers = bins[extreme_max_idx]
    boundaries = np.hstack((np.array([-np.inf]), bins[extreme_min_idx], np.array([np.inf])))

    clusters_idx = []
    for center in centers:
        # 获取包围簇心的左右边界
        left_boundary = boundaries[np.searchsorted(boundaries, center) - 1]
        right_boundary = boundaries[np.searchsorted(boundaries, center)]

        cluster_idx = np.argwhere((left_boundary <= array) & (array < right_boundary)).reshape(-1)

        # 过滤偏差较小的簇
        cluster = array[cluster_idx]
        mu = np.mean(np.abs(cluster))
        if np.abs(mu) < max_normal_deviation or len(cluster) == 0:
            continue

        clusters_idx.append(cluster_idx)
    return np.array(clusters_idx)
