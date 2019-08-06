"""
实现An Entropy Weighting k-Means Algorithm for Subspace
Clustering of High-Dimensional Sparse Data

@author:lyh
"""

import math
import random
import numpy as np

def init_cluster_center(data, cluster_number):
    """
    初始化簇中心,从数据中随机选择cluster_number个作为簇中心，然后返回

    @param data:np.array
    @param cluster_number:簇的个数
    """
    n = np.size(data, 0) # 数据大小
    rand_index = np.array(random.sample(range(1, n), cluster_number))
    cluster_centers = data[rand_index, :]
    return cluster_centers

def find_closest_cluster_center(data, weight, cluster_centers):
    """
    查找data中每个数据的最近的簇，返回所有簇索引

    @param data: numpy.array
    @param wight: 权重
    @param cluster_centers: 所有簇中心
    """
    cluster_number = np.size(cluster_centers, axis=0) # 多少簇(行数)
    n,m = data.shape
    cluster_index = np.zeros((np.size(data, axis=0)), dtype=int)
    for i in range(n):
        distance = np.power((cluster_centers - data[i, :]), 2)   # numpy.power分别求每个元素的平方
        weight_distance = np.multiply(distance, weight)       # numpy.multiply对应元素相乘
        weight_distance_sum = np.sum(weight_distance, axis=1) # 行相加
        if math.isinf(weight_distance_sum.sum()) or math.isnan(weight_distance_sum.sum()):
            weight_distance_sum = np.zeros(cluster_number)
        # 得到索引
        cluster_index[i] = np.where(weight_distance_sum == weight_distance_sum.min())[0][0]
    return cluster_index

def compute_cluster_center(data, cluster_index, cluster_number):
    """
    计算簇中心,return centers of all cluster

    @param data: numpy.array
    @param cluster_index: data属于ith簇的索引
    @param cluster_number: 簇个数
    """
    n,m = data.shape
    cluster_centers = np.zeros((cluster_number, m), dtype=float)
    for k in range(cluster_number):
        # 对于每个簇分别计算
        index = np.where(cluster_index == k)[0]
        temp = data[index, :]                 # 取出簇中所有数据
        all_dimen_sum = np.sum(temp, axis=0)  # 按列相加,分别求所有维度的总和
        cluster_centers[k, :] = all_dimen_sum / np.size(index)
    return cluster_centers

def compute_weight(data, cluster_centers, cluster_index, gamma):
    """
    计算权重

    @form: λlt = (exp(-Dlt/γ)) / (Σ(1->M)exp(-Dli/γ)), Dlt = Σ(1->n) wlj(zlt - xjt)^2
    @param data: numpy.array
    @param cluster_centers: 簇中心
    @param gamma: 控制多维子空间聚类激励强度的正参数
    """
    cluster_number = np.size(cluster_centers, 0)
    n,m = data.shape
    weight = np.zeros((cluster_number, m))
    distance_sum = np.zeros((cluster_number, m), dtype=float)
    for k in range(cluster_number):
        index = np.where(cluster_index == k)[0]
        temp = data[index, :]                 # 取出簇中所有数据
        distance = np.power((temp - cluster_centers[k, :]), 2)
        distance_sum[k, :] = np.sum(distance, axis=0) # 按列相加
    for k in range(cluster_number):
        numerator = np.exp(np.divide(-distance_sum[k, :], gamma))
        denominator = np.sum(numerator, axis=0)  # 按列相加
        weight[k, :] = np.divide(numerator, denominator)
    return weight

def cost_function(data, cluster_centers, cluster_index, weight, gamma):
    """
    目标函数

    @param data: numpy.array
    @param cluster_centers: 簇中心
    @param cluster_index: 簇索引
    @param weight: 权重
    @param gamma:激励强度的参数
    """
    cost = 0
    cluster_number = np.size(cluster_centers, 0)
    n,m = data.shape
    for k in range(cluster_number):
        index = np.where(cluster_index == k)[0]
        temp = data[index, :]
        distance = np.power((temp - cluster_centers[k, :]), 2)
        weight_distance = np.multiply(distance, weight[k, :])
        temp = gamma*np.dot(weight[k, :], np.log(weight[k, :]))
        cost = cost + np.sum(weight_distance) + temp
    return cost

def is_convergence(cost_func):
    """
    判断函数是否收敛
    """
    result = True
    if math.isnan(np.sum(cost_func)):
        result = False
    iteration = np.size(cost_func)
    for i in range(iteration-1):
        if cost_func[i] < cost_func[i+1]:
            result = False
        i += 1
    return result

def ewkmeans(data, cluster_number, gamma, iterations):
    """
    EWKM 的实现

    @param data: numpy.array
    @param cluster_number: 簇数量
    @param gamma: γ参数
    @param iterations: 迭代次数
    """
    n,m = data.shape
    cost_func = np.zeros(iterations)
    # 初始化权重
    weight = np.zeros((cluster_number, m), dtype=float) + np.divide(1, float(m))
    # 初始化簇中心
    cluster_centers = init_cluster_center(data, cluster_number)
    for i in range(iterations):
        cluster_index = find_closest_cluster_center(data, weight, cluster_centers)
        cluster_centers = compute_cluster_center(data, cluster_index, cluster_number)
        weight = compute_weight(data, cluster_centers, cluster_index, gamma)
        cost_func[i] = cost_function(data, cluster_centers, cluster_index, weight, gamma)
    best_labels = cluster_index
    best_centers = cluster_centers
    if not (math.isnan(np.sum(cost_func)))  and is_convergence(cost_func):
        return True, best_labels, best_centers
    else:
        return False, None, None


class EWKmeans:
    n_cluster = 0
    max_iter = 0
    gamma = 0
    best_labels, best_centers = None, None
    is_converge = False

    def __init__(self, n_cluster=3, max_iter=20, gamma=10.0):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.gamma = gamma


    def fit(self, data):
        self.is_converge, self.best_labels, self.best_centers = ewkmeans(
            data=data, cluster_number=self.n_cluster, gamma=self.gamma, iterations=self.max_iter
        )
        return self

    def fit_predict(self, data):
        if self.fit(data).is_converge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'

    def get_params(self):
        return self.is_converge, self.n_cluster, self.gamma

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    DATA = load_iris()
    TRAIN_DATA = DATA.data
    TRAIN_TARGET = DATA.target
    print(TRAIN_TARGET)
    print(TRAIN_DATA)
    EWK = EWKmeans(n_cluster=3, gamma=10.0)
    Y_PRE = EWK.fit_predict(TRAIN_DATA)
    print(Y_PRE)
