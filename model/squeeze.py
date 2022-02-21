from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cityblock

from model.anomaly_fileter import filter_based_deviation
from model.density_cluster import cluster_based_density
from utils.attribute_combination import AttributeCombination as AC


class Squeeze:
    def __init__(self, data, param):
        self.param = param
        self.normal_idx = None
        self.clusters = None
        self._root_cause = []
        self.anomaly_idx = None

        # 过滤掉真实值与预测值全为0的数据
        data = data[((data.real > 0) | (data.predict > 0))]
        self.data = data[((data.real < np.inf) & (data.predict < np.inf))]
        self._v = self.data['real'].values
        self._f = self.data['predict'].values

        # 计算偏差得分
        self.deviation_scores = 2 * (self._f - self._v) / (self._f + self._v)

        # 获取 attributes, set(): 列名可能重复; sorted():因为AC里sorted()了,这里如果不,顺序会不一致
        self.attribute_names = list(sorted(set(self.data.columns) - {'real', 'predict'}))

        # 获取每个 attributes 的 values
        self.attribute_values = list(list(set(self.data.loc[:, name].values)) for name in self.attribute_names)

        # dataframe -> {column: value} -> AC
        self.ac_array = np.array(
            [AC(**record) for record in self.data[self.attribute_names].to_dict(orient='records')])

    @property
    @lru_cache()
    def root_cause(self):
        return self._root_cause

    @lru_cache()
    def get_cuboid_ac_array(self, cuboid):
        return np.array(list(map(lambda x: x.mask(cuboid), self.ac_array)))

    @lru_cache()
    def get_indexed_data(self, cuboid):
        return self.data.set_index(list(cuboid))

    def run(self):
        # 基于偏差的过滤
        self.anomaly_idx = filter_based_deviation(self._v, self._f)

        # 基于偏差得分的聚类
        clusters = cluster_based_density(self.deviation_scores[self.anomaly_idx], self.param.max_normal_deviation)
        clusters = list([self.anomaly_idx[_] for _ in clusters])
        self.clusters = clusters
        if len(self.clusters) == 0:
            return

        self.anomaly_idx = np.concatenate(self.clusters)
        self.normal_idx = np.argwhere(np.abs(self.deviation_scores) <
                                      np.min(np.abs(self.deviation_scores[self.anomaly_idx]))).reshape(-1)

        # 计算常数C用于平衡GPS和简洁性
        self.param.score_weight = - np.log(
            len(self.clusters) *
            sum(len(_) for _ in self.clusters) / len(self._f)) / np.log(
            sum(len(_) for _ in self.attribute_values)) * sum(len(_) for _ in self.attribute_values)

        for cluster in self.clusters:
            self._locate_in_cluster(cluster)

    def _locate_in_cluster(self, cluster):
        ret_lists = []
        for cuboid_layer in range(1, len(self.attribute_names) + 1):
            # map(fun, list): 对整个 list 应用 fun 函数
            # combinations(list, n): 返回 list 中长度为 n 的子列表
            # 获取每个cuboid的最佳分区和最高得分
            layer_ret_lists = list(map(
                lambda x: self._locate_in_cuboid(x, cluster=cluster),
                combinations(self.attribute_names, cuboid_layer)
            ))

            ret_lists.extend([
                {'rc': x[0], 'score': x[1], 'n_ele': len(x[0]), 'layer': cuboid_layer,
                 'rank': x[1] * self.param.score_weight - len(x[0]) * cuboid_layer
                 } for x in layer_ret_lists
            ])

            # 如果当前层搜索到得分大于阈值的根因，停止搜索
            if len(list(filter(lambda x: x['score'] > self.param.ps_upper_bound, ret_lists))):
                break

        # 对所有搜索到的根因综合得分与简洁进行降序排列
        ret_lists = list(sorted(
            ret_lists,
            key=lambda x: x['rank'],
            reverse=True)
        )

        # 添加排名第一的为根因
        if ret_lists:
            ret = ret_lists[0]['rc']
            self._root_cause.append(frozenset(ret))

    def _locate_in_cuboid(self, cuboid, cluster):
        # 将 cuboid 作为 dataframe 的索引
        data_cuboid_indexed = self.get_indexed_data(cuboid)

        # 对于簇中元素屏蔽掉 cuboid 之外的其他 attribute
        abnormal_cuboid_ac_arr = self.get_cuboid_ac_array(cuboid)[cluster]

        # 对于簇中数据获取 cuboid 里的 attribute values 的种类和个数
        elements, num_elements = np.unique(abnormal_cuboid_ac_arr, return_counts=True)

        # 对于所有数据获取cuboid里的attribute values的种类和个数
        num_ele_descents = np.array(list(
            np.count_nonzero(
                _.index_dataframe(data_cuboid_indexed),
            ) for _ in elements
        ))

        # 获得descent score来表示后代叶子属性组合在异常簇中的比例
        descent_score = num_elements / num_ele_descents

        # 按照descent score降序排列
        idx = np.argsort(descent_score)[::-1]
        elements = elements[idx]

        # partitions: 需要考虑的elements个数，即异常簇的cuboid里的attribute values的descent score的前partitions个，  种类
        partitions = np.arange(
            min(
                len(elements),
                self.param.max_num_elements_single_cluster
            )
        ) + 1

        # 获得每个分区的GPS分数
        rc_scores = np.asarray(
            list(map(lambda x: self._root_cause_score(cluster=cluster, cuboid=cuboid, elements=elements, partition=x),
                     partitions)))

        # 返回得分最高的分区和得分
        idx = np.argsort(rc_scores)[::-1]
        partitions = partitions[idx]
        rc_scores = rc_scores[idx]
        score = rc_scores[0]
        rc = elements[:partitions[0].item()]

        return rc.tolist(), score

    def get_derived_dataframe(self, ac_set, cuboid, subset_indices):
        # subset: 正常元素 + 该簇中的异常元素
        # idx: 满足当前elements切片的元素
        subset = np.zeros(len(self.data), dtype=np.bool)
        subset[subset_indices] = True
        idx = AC.batch_index_dataframe(ac_set, self.get_indexed_data(cuboid))

        data = self.data[idx & subset]
        complement_data = self.data[(~idx) & subset]
        return data, complement_data

    def _root_cause_score(self, cluster, cuboid, elements, partition):
        # data_p: [正常元素 + 该簇中的异常元素] & 满足当前 elements 的切片 S1
        # data_n: [正常元素 + 该簇中的异常元素] & 不满足当前 elements 的切片 S2
        """
        与论文不符?
        """
        data_p, data_n = self.get_derived_dataframe(
            (elements[:partition]), cuboid=cuboid,
            subset_indices=np.sort(np.concatenate([cluster, self.normal_idx])))

        _v1, _v2 = data_p.real.values, data_n.real.values
        _f1, _f2 = data_p.predict.values, data_n.predict.values

        _pv, _pf = np.sum(data_p.real.values), np.sum(data_p.predict.values)
        _a1, _a2 = data_p.predict.values * (_pv / _pf), data_n.predict.values

        # L1范数即绝对值之和
        _ps = 1 - ((np.mean(cityblock(_v1, _a1)) + np.mean(cityblock(_v2, _f2)))
                   / (np.mean(cityblock(_v1, _f1)) + np.mean(cityblock(_v2, _f2))))

        return _ps
