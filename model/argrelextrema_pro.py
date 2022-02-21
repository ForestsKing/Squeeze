import numpy as np
from scipy.signal import argrelextrema


def argrelmax_pro(array):
    new_array_idx = [0]
    new_array = [array[0]]

    for i in range(len(array) - 1):
        if array[i + 1] != array[i]:
            new_array_idx.append(i + 1)
            new_array.append(array[i + 1])

    new_array_idx = np.array(new_array_idx)
    new_array = np.array(new_array)

    relmax_idx = argrelextrema(np.array(new_array), np.greater_equal)
    return new_array_idx[relmax_idx]


def argrelmin_pro(array):
    new_array_idx = [0]
    new_array = [array[0]]

    for i in range(len(array) - 1):
        if array[i + 1] != array[i]:
            new_array_idx.append(i + 1)
            new_array.append(array[i + 1])

    new_array_idx = np.array(new_array_idx)
    new_array = np.array(new_array)

    relmin_idx = argrelextrema(np.array(new_array), np.less_equal)
    return new_array_idx[relmin_idx]
