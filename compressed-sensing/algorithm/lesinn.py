import numpy as np
#import cupy as cp
import random
from time import perf_counter
import os


lesinn_cuda_kernel_raw = ""
with open(os.getcwd()+"/compressed-sensing/algorithm/cuda/lesinn.cu") as f:
    lesinn_cuda_kernel_raw = f.read()

#lesinn_cuda_kernel = cp.RawKernel(lesinn_cuda_kernel_raw, 'lesinn')


def similarity(x: np.ndarray, y: np.ndarray):
    """
    计算两个向量的相似度
    :param x:
    :param y:
    :return:
    """
    a = 1 / (1 + np.sqrt(np.sum(np.square(x - y))))
    return a

#这个 online_lesinn 函数实现了 LESINN 算法，通过在历史数据和待计算数据之间构建子集，并计算子集中样本与待计算数据的相似度来评估离群值分数。
# 最后返回每个数据点的离群值分数。
def online_lesinn(
        incoming_data: np.array,
        historical_data: np.array,
        t: int = 50,
        phi: int = 20,
        random_state: int = None
):
    """
    在线离群值算法 lesinn
    :param incoming_data: shape=(m, d,) 需要计算离群值的向量
    :param historical_data: shape=(n,d) 历史数据
    :param t: 每个数据点取t个data中子集作为离群值参考
    :param phi: 每个数据t个子集的大小
    :param random_state: 随机数种子
    :return:
    """
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    m = incoming_data.shape[0]
    # 将历史所有数据和需要计算离群值的数据拼接到一起
    if historical_data.shape:
        all_data = np.concatenate([historical_data, incoming_data], axis=0)
    else:
        all_data = incoming_data
    n, d = all_data.shape
    data_score = np.zeros((m,))

    # incoming_data = cp.array(incoming_data, order='C')
    # all_data = cp.array(all_data, order='C')

    # lesinn_cuda_kernel(
    #     (1,), (m,), (incoming_data, all_data, m, n, d, t, phi, data_score))
    # cp.cuda.runtime.deviceSynchronize()
    # return data_score

    for i in range(m):
        score = 0
        for j in range(t):
            sample = random.sample(range(0, n), k=phi)
            nn_sim = 0
            for each in sample:
                nn_sim = max(
                    nn_sim, similarity(incoming_data[i, :], all_data[each, :])
                )
            score += nn_sim
        if score:
            data_score[i] = t / score
    return data_score

