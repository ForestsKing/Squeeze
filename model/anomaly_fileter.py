import numpy as np
from kneed import KneeLocator
from scipy.stats import gaussian_kde


def filter_based_deviation(real_array, predict_array):
    # 计算偏差
    deviation = np.abs(real_array - predict_array)
    # 获取每个偏差概率密度
    kernel = gaussian_kde(deviation)

    # 获取小于每个偏差概率
    _x = np.linspace(np.min(deviation), np.max(deviation), 1000)
    _y = np.cumsum(kernel(_x))

    # 获取概率的拐点
    knee = KneeLocator(_x, _y, curve='concave', direction='increasing').knee
    if knee is None:
        knee = np.min(deviation)

    # 获取偏差大于拐点的索引，也就是获取偏差较为明显的索引
    indices = np.argwhere(deviation > knee).reshape(-1)
    return indices
