import os
import time

import math
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression


class Sampler(object):
    def __init__(self, err_tol, conf, Y_pred, Y_true, R):
        # print('[Attention ! here is Y_Pre]: ',Y_pred)
        # print('[Attention ! here is type(Y_Pre)]: ',type(Y_pred))
        # print('[Attention ! here is len(Y_Pre)]: ',len(Y_pred))
        # output_dir = os.path.dirname('/home/wangshuo_20/pythonpr/seiden_ws/benchmarks/stanford/blazeit/blazeit/out/Y_pred_TASTI.txt')
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # np.savetxt('/home/wangshuo_20/pythonpr/seiden_ws/benchmarks/stanford/blazeit/blazeit/out/Y_pred_TASTI.txt', Y_pred)
        self.ensure_data_legitimacy(Y_pred)
        self.err_tol = err_tol
        self.conf = conf
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        self.R = R

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


    def get_sample(self, Y_pred, Y_true, nb_samples):
        raise NotImplementedError

    def reset(self, Y_pred, Y_true):
        pass

    def reestimate(self, Y_pred, Y_true, nb_samples):
        return None, None

    def permute(self, Y_pred, Y_true):
        p = np.random.permutation(len(Y_pred))
        Y_pred, Y_true = Y_pred[p], Y_true[p]
        return Y_pred, Y_true

    def sample(self):
        stp = time.perf_counter()
        # print('[Here start running self.Y_true.astype]')
        # Y_pred, Y_true = self.permute(self.Y_pred.astype(np.float32), self.Y_true.astype(np.float32))
        Y_pred, Y_true = self.permute(self.Y_pred.astype(np.float32), self.Y_true)
        etp = time.perf_counter()
        # print('[Here is premute time]：',etp-stp)
        # print('[Here is y_true]： ',Y_true)
        # print('[Here is y_pre]： ',Y_pred)
        # for it in Y_pred:
        #     print('[y_pred]: ',it)
        self.reset(Y_pred, Y_true)
        LB = 0
        UB = 10000000
        t = 1
        k = 1
        beta = 1.5
        R = self.R
        eps = self.err_tol
        p = 1.1
        c = self.conf * (p - 1) / p
        # print("[R]: ", R)

        st = time.perf_counter()
        Xt_sum = self.get_sample(Y_pred, Y_true, t)
        Xt_sqsum = Xt_sum * Xt_sum
        # print('[Xt_sum]:',Xt_sum)
        # while (1 + eps) * LB < (1 - eps) * UB:
        while LB + eps < UB - eps:
            # print('[Circle_Count] 3', t)
            # print('[UB]', UB)
            # print('[LB]', LB)
            if (t > len(Y_true)):
                print("[Miss Convergence]")
                break
            t += 1
            if t > np.floor(beta ** k):
                k += 1
                alpha = np.floor(beta ** k) / np.floor(beta ** (k - 1))
                dk = c / (math.log(k, p) ** p)
                x = -alpha * np.log(dk) / 3
                # print(fr't: {t},k: {k}, x: {x}')
                t1, t2 = self.reestimate(Y_pred, Y_true, t)
                if t1 is not None and t2 is not None:
                    Xt_sum = t1
                    Xt_sqsum = t2

            sample = self.get_sample(Y_pred, Y_true, t)
            # print('[Sample]: ', sample)
            Xt_sum += sample
            Xt_sqsum += sample * sample
            Xt = Xt_sum / t
            sigmat = np.sqrt(1 / t * (Xt_sqsum - Xt_sum ** 2 / t))
            # print('[sigmat]: ',sigmat)
            # Finite sample correction
            sigmat *= np.sqrt((len(Y_true) - t) / (len(Y_true) - 1))

            ct = sigmat * np.sqrt(2 * x / t) + 3 * R * x / t
            # print('[Xt+ct]: ',np.abs(Xt) + ct)
            # print('[Xt]: ',np.abs(Xt))
            # print('[ct]: ',ct)
            LB = max(LB, np.abs(Xt) - ct)
            UB = min(UB, np.abs(Xt) + ct)
        et = time.perf_counter()
        print('[Here is UB,LB iterator times]:' ,t)
        estimate = np.sign(Xt) * 0.5 * \
                   ((1 + eps) * LB + (1 - eps) * UB)
        return estimate * len(Y_true), t


class TrueSampler(Sampler):
    def get_sample(self, Y_pred, Y_true, nb_samples):
        return Y_true[nb_samples]


class ControlCovariateSampler(Sampler):
    def __init__(self, *args):
        super().__init__(*args)
        self.tau = np.mean(self.Y_pred)
        self.var_t = np.var(self.Y_pred)
        print('[ self.tau ]: ', self.tau )
        print('[ self.var_t ]: ', self.var_t  )

    def reset(self, Y_pred, Y_true):
        self.cov = np.cov(Y_true[0:100].astype(np.float32), Y_pred[0:100].astype(np.float32))[0][1]
        self.c = -1 * self.cov / self.var_t

    def reestimate(self, Y_pred, Y_true, nb_samples):
        # yt_samp = Y_true[0:nb_samples]
        # yp_samp = Y_pred[0:nb_samples]
        yt_samp = Y_true[0:nb_samples].astype(np.float32)
        yp_samp = Y_pred[0:nb_samples].astype(np.float32)
        self.cov = np.cov(yt_samp, yp_samp)[0][1]
        self.c = -1 * self.cov / self.var_t

        samples = yt_samp + self.c * (yp_samp - self.tau)
        Xt_sum = np.sum(samples)
        Xt_sqsum = sum([x * x for x in samples])
        return Xt_sum, Xt_sqsum

    def _get_yp(self, Y_true, Y_pred, nb_samples):
        return Y_pred[nb_samples]

    def get_sample(self, Y_pred, Y_true, nb_samples):
        if nb_samples >= len(Y_true):
            nb_samples = len(Y_true) - 1
        yt_samp = Y_true[nb_samples]
        yp_samp = self._get_yp(Y_true, Y_pred, nb_samples)
        # m^ =  m = c*(a- E(a))
        sample = yt_samp + self.c * (yp_samp - self.tau)
        return sample


class MultiControlCovariateSampler(ControlCovariateSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def reset(self, Y_pred, Y_true):
        self.reg = LinearRegression().fit(Y_pred[0:100], Y_true[0:100])
        yp = self.reg.predict(Y_pred[0:100])
        super().reset(yp, Y_true)

    def reestimate(self, Y_pred, Y_true, nb_samples):
        self.reg = LinearRegression().fit(Y_pred[0:nb_samples], Y_true[0:nb_samples])
        yp = self.reg.predict(Y_pred[0:nb_samples])
        return super().reestimate(yp, Y_true, nb_samples)

    def _get_yp(self, Y_true, Y_pred, nb_samples):
        return self.reg.predict(Y_pred[nb_samples:nb_samples + 1])[0]
