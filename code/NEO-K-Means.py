"""
实现NEO-K-Means 算法

:author:lyh
"""

import math
import random
import numpy as np


def init_cluster_center(data, cluster_number):
    """
    初始化簇中心,从数据中随机选择cluster_number个作为簇中心，然后返回

    :param data:np.array
    :param cluster_number:簇的个数
    """
    n = np.size(data, 0) # 数据大小
    rand_index = np.array(random.sample(range(1, n), cluster_number))
    cluster_centers = data[rand_index, :]
    return cluster_centers


def get_dp_center_from_dist(dist_mtx, Tao, S, stage=1):
    """
    从距离矩阵中找到数据点到簇中心最小值

    :param dist_mtx: numpy.array
    :param Tao: tao集合
    :param S: S集合
    :param stage: 第1阶段还是第2阶段
    :return: 元组
    """
    i=0; j=0
    temp=float('inf')
    N, K = np.shape(dist_mtx)
    if stage == 1:
        for i_temp in range(N):
            for j_temp in range(K):
                if (dist_mtx[i_temp, j_temp] < temp) and ((i_temp, j_temp) not in Tao) and (i_temp not in S):
                    i = i_temp
                    j = j_temp
    else:
        for i_temp in range(N):
            for j_temp in range(K):
                if (dist_mtx[i_temp, j_temp] < temp) and ((i_temp, j_temp) not in Tao):
                    i = i_temp
                    j = j_temp
    return i, j


def compute_cluster_center(data, indicator_mtx):
    """
    计算簇中心,return centers of all cluster

    :param data: numpy.array
    :param indicator_mtx: 指示矩阵
    :return: cluster_center
    """
    K = np.size(indicator_mtx, axis=1)
    M = np.size(data, axis=1)
    cluster_centers = np.zeros(shape=(K,M), dtype=float)
    for k in range(K):
        # 对于每个簇分别计算
        temp = np.dot(indicator_mtx[:, k], data)
        non_zero = np.sum(indicator_mtx[:, k])
        cluster_centers[k] = temp/non_zero
    return cluster_centers


def compute_weight(data, cluster_centers, cluster_index, gamma):
    """
    计算权重

    :form: λlt = (exp(-Dlt/γ)) / (Σ(1->M)exp(-Dli/γ)), Dlt = Σ(1->n) wlj(zlt - xjt)^2
    :param data: numpy.array
    :param cluster_centers: 簇中心
    :param gamma: 控制多维子空间聚类激励强度的正参数
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

def object_funtion(data, cluster_centers, indicator_mtx):
    """
    目标函数

    :param data: numpy.array
    :param cluster_centers: 簇中心
    :param indicator_mtx: 指示矩阵
    :return
    """
    cost = 0
    N, K = np.shape(indicator_mtx)
    for i in range(N):
        for j in range(K):
            if indicator_mtx[i, j] == 1:
                cost = cost + np.linalg.norm(data[i] - cluster_centers[j])
    return cost

def is_convergence(cost_func):
    """
    判断函数是否收敛
    """
    result = True
    if math.isnan(np.sum(cost_func)): # 如果 x = Non (not a number) 返回真
        result = False
    iteration = np.size(cost_func)
    for i in range(iteration-1):
        if cost_func[i] < cost_func[i+1]:
            result = False
        i += 1
    return result


def get_distance_mtx(data, cluster_center):
    """
    计算数据点与簇中心距离矩阵
    :param data: numpy.array N*M(N个数据点, M个维度)
    :param cluster_center: numpy.array, 簇中心所组成的矩阵 K*M(K个簇中心, M个维度)
    :return: numpy.array 数据点与簇中心的距离矩阵 N*K(N个数据点, K个簇中心)
    """
    N = np.size(data, axis=0)  # axis为0, 取行数
    K = np.size(cluster_center, axis=0)  # axis为1, 取列数
    dist_mtx = np.zeros(shape=(N, K), dtype=float)
    M = np.size(data, axis=1)
    for i in range(N):
        for j in range(K):
            dist_mtx[i, j] = np.linalg.norm(data[i] - cluster_center[j])
    return dist_mtx


def neokmeans(data, cluster_number, alpha, beta, iterations):
    """
    NEO-K-Means 的实现

    :param data: numpy.array
    :param cluster_number: 簇数量
    :param alpha: alpha参数 0 < alpha << (k-1)
    :param beta: beta参数 0 < beta*N << N
    :param iterations: 迭代次数
    """
    N = np.size(data, axis=0)
    cost_func = np.zeros(iterations)
    # 初始化权重
    # weight = np.zeros((cluster_number, m), dtype=float) + np.divide(1, float(m))
    # 初始化簇中心
    cluster_centers = init_cluster_center(data, cluster_number)
    # 得到距离矩阵
    converged = False
    t = 0
    cost_func[0] = float('inf')
    indicator_mtx = np.zeros(shape=(N, cluster_number), dtype=float)
    while not converged and (t < iterations):
        dist_mtx = get_distance_mtx(data, cluster_centers)
        Tao = set(); S = set(); p = 0
        indicator_mtx = np.zeros(shape=(N, cluster_number), dtype=float)
        while p < (N+alpha*N):
            i=0; j=0
            if p < (N-beta*N):
                i, j = get_dp_center_from_dist(dist_mtx, Tao, S, stage=1)
                S.add(i)
            else:
                i, j = get_dp_center_from_dist(dist_mtx, Tao, S, stage=2)

            Tao.add((i, j))
            p = p + 1
            indicator_mtx[i, j] = 1
        cluster_centers = compute_cluster_center(data, indicator_mtx)
        of = object_funtion(data, cluster_centers, indicator_mtx)
        if of - cost_func[t] < 0.01:
            converged = True
        t = t + 1
    return indicator_mtx

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    DATA = load_iris()
    data = DATA.data
    im = neokmeans(data, 3, 0.01, 0.1, 300)
    print(im)
