from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler, SamplingBounds
from supg.selector.base_selector import BaseSelector, ApproxQuery


class ImportancePrecisionTwoStageSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: ImportanceSampler,
        start_samp=100,
        step_size=100

    ):
        self.query = query
        self.data = data
        print('[two_Import]')
        self.ensure_data_legitimacy(self.data.get_y_pred())
        self.sampler = sampler
        if not isinstance(sampler, ImportanceSampler):
            raise Exception("Invalid sampler for importance")
        self.start_samp = start_samp
        self.step_size = step_size
    def ensure_data_legitimacy(self,data):
        # 检查是否有空值（np.nan）或非法值
        nan_mask = np.isnan(data)  # 检查空值（np.nan）
        inf_mask = np.isinf(data)  # 检查无穷大值（np.inf 或 -np.inf）
        # 获取所有负数的索引
        negative_indices = np.where(data < 0)[0]
        if len(negative_indices) > 0:
            print("数组中存在负数，负数索引为:", negative_indices)
        else:
            print("数组中不存在负数.")
        # 打印包含空值或非法值的索引
        nan_indices = np.where(nan_mask)[0]
        inf_indices = np.where(inf_mask)[0]
        print("包含空值（np.nan）的索引：", nan_indices)
        print("包含无穷大值（np.inf 或 -np.inf）的索引：", inf_indices)
        data[nan_mask] = 0  # 将空值替换为0
        data[inf_mask] = 0  # 将无穷大值替换为0
    def select(self) -> Sequence:
        # self.ensure_data_legitimacy(self.data.get_y_pred())
        print(fr'[start_samp]: {self.start_samp}')
        data_idxs = self.data.get_ordered_idxs()
        n = len(data_idxs)
        T = 1 + 2 * (self.query.budget - self.start_samp) // self.step_size
        # TODO: weights
        x_probs = self.data.get_y_prob()
        # self.sampler.set_weights(np.repeat(1.,n)/n)
        self.sampler.set_weights(np.sqrt(x_probs))
        # self.sampler.set_weights(x_probs ** 2)
        # self.sampler.set_weights(x_probs)

        x_ranks = np.arange(n)
        x_basep = np.repeat((1./n),n)
        x_weights = self.sampler.weights

        n_sample_1 = self.query.budget // 2
        n_sample_2 = self.query.budget - n_sample_1

        samp_ranks = np.sort(self.sampler.sample(max_idx=n, s=n_sample_1))
        samp_basep = x_basep[samp_ranks]
        samp_weights = x_weights[samp_ranks]
        samp_ids = data_idxs[samp_ranks]
        samp_labels = self.data.lookup(samp_ids)
        samp_masses = samp_basep / samp_weights

        delta = self.query.delta
        bounder = SamplingBounds(delta=delta / T)
        tpr_lb, tpr_ub = bounder.calc_bounds(
            fx = samp_labels*samp_masses,
        )
        cutoff_ub = int(math.ceil(tpr_ub * n / self.query.min_precision))
        # print('cutoff ub: {}'.format(cutoff_ub))

        samp2_ranks = np.sort(self.sampler.sample(max_idx=cutoff_ub, s=n_sample_2))
        x_weights = self.sampler.weights
        samp2_basep = x_basep[samp2_ranks]
        samp2_weights = x_weights[samp2_ranks]
        samp2_ids = data_idxs[samp2_ranks]
        samp2_labels = self.data.lookup(samp2_ids)
        samp2_masses = samp2_basep / samp2_weights
        # print("ns2: {}, len(samp2): {}".format(n_sample_2, len(samp2_ids)))

        allowed = [0]
        for s_idx in range(self.start_samp, n_sample_2, self.step_size):
            if s_idx + 1 >= len(samp2_ranks):
                continue
            cur_u_idx = samp2_ranks[s_idx]
            # print("curidx: {}, s_idx: {}".format(cur_u_idx, s_idx))
            cur_x_basep = x_basep[:cur_u_idx+1] / np.sum(x_basep[:cur_u_idx+1])
            cur_x_weights = x_weights[:cur_u_idx+1] / np.sum(x_weights[:cur_u_idx+1])
            cur_x_masses = cur_x_basep / cur_x_weights

            cur_subsample_x_idxs = samp2_ranks[:s_idx+1]

            pos_rank_lb, pos_rank_ub = bounder.calc_bounds(
                fx=samp2_labels[:s_idx+1]*cur_x_masses[cur_subsample_x_idxs],
            )
            prec_lb = pos_rank_lb
            if prec_lb > self.query.min_precision:
                allowed.append(cur_u_idx)

        set_inds = data_idxs[:allowed[-1]]
        samp_inds = self.data.filter(np.concatenate([samp_ids, samp2_ids]))
        return np.unique(np.concatenate([set_inds, samp_inds]))