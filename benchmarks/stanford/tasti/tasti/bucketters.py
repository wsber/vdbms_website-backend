import numpy as np
import tqdm
from numba import njit, prange


@njit(parallel=True)
def get_dists(x, embeddings):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
    return dists


@njit(parallel=True)
def get_and_update_dists(x, embeddings, min_dists):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
        if dists[i] < min_dists[i]:
            min_dists[i] = dists[i]
    return dists


class Bucketter(object):
    def __init__(
            self,
            nb_buckets: int,
            seed: int = 123456
    ):
        self.nb_buckets = nb_buckets
        self.rand = np.random.RandomState(seed)

    def topk(self, k, dists):
        topks = []
        for i in tqdm.tqdm(range(len(dists))):
            topks.append(np.argpartition(dists[i], k)[0:k])
        return np.array(topks)


class FPFRandomBucketter(Bucketter):
    def bucket(
            self,
            embeddings: np.ndarray,
            max_k: int,
            percent_fpf=0.75
    ):

        reps = np.full(self.nb_buckets, -1)
        print('[FPF_REPS]: ',len(reps))
        reps[0] = self.rand.randint(len(embeddings))
        min_dists = np.full(len(embeddings), np.Inf, dtype=np.float32)
        dists = np.zeros((self.nb_buckets, len(embeddings)))
        dists[0, :] = get_and_update_dists(
            embeddings[reps[0]],
            embeddings,
            min_dists
        )

        num_random = int((1 - percent_fpf) * len(reps))
        print('[num_random]: ', num_random)
        # 随机选择剩余的代表点，并更新距离矩阵和最小距离数组
        for i in tqdm.tqdm(range(1, num_random), desc='RandomBucketter'):
            reps[i] = self.rand.randint(len(embeddings))
            # print('[rep_random]: ', reps[i])
            dists[i, :] = get_and_update_dists(
                embeddings[reps[i]],
                embeddings,
                min_dists
            )
        # for i in range(len(min_dists)):
        #     print('min_dist: ', min_dists[i])
        # 使用 FPF 策略选择余下的代表点，并更新距离矩阵和最小距离数组
        for i in tqdm.tqdm(range(num_random, self.nb_buckets), desc='FPFBucketter'):
            reps[i] = np.argmax(min_dists)
            # print('[rep_fpf]: ', reps[i])
            dists[i, :] = get_and_update_dists(
                embeddings[reps[i]],
                embeddings,
                min_dists
            )
        # 转置距离矩阵，以便每行表示一个数据点到所有代表点的距离
        dists = dists.transpose()
        # 获取每行的最小距离的前 k 个索引
        topk_reps = self.topk(max_k, dists)
        # 初始化最小距离矩阵，用于存储每个数据点到最近的 k 个代表点的距离
        topk_dists = np.zeros([len(topk_reps), max_k])

        # 获取每个数据点到最近的 k 个代表点的距离
        for i in range(len(topk_dists)):
            topk_dists[i] = dists[i, topk_reps[i]]
        # 获取最近的 k 个代表点的索引
        for i in range(len(topk_reps)):
            topk_reps[i] = reps[topk_reps[i]]
        # 返回最终的代表点数组、最近的 k 个代表点的索引数组和最近的 k 个代表点的距离矩阵
        return reps, topk_reps, topk_dists


class CrackingBucketter(Bucketter):
    def bucket(
            self,
            embeddings: np.ndarray,
            max_k: int,
            idxs: list
    ):
        reps = idxs
        min_dists = np.full(len(embeddings), np.Inf, dtype=np.float32)
        dists = np.zeros((len(reps), len(embeddings)), dtype=np.float32)
        for i in tqdm.tqdm(range(len(reps)), desc='Cracking'):
            dists[i, :] = get_and_update_dists(
                embeddings[reps[i]],
                embeddings,
                min_dists
            )
        dists = dists.transpose()
        topk_reps = self.topk(max_k, dists)
        topk_dists = np.zeros([len(topk_reps), max_k])

        for i in range(len(topk_dists)):
            topk_dists[i] = dists[i, topk_reps[i]]
        for i in range(len(topk_reps)):
            topk_reps[i] = reps[topk_reps[i]]

        return reps, topk_reps, topk_dists
