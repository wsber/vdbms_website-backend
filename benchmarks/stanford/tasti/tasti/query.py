import tasti
import sklearn
import numpy as np
import pandas as pd
import supg.datasource as datasource
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler, TrueSampler
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector, ImportancePrecisionTwoStageSelector
from tabulate import tabulate
import torch
import torchvision
import time


def print_dict(d, header='Key'):
    headers = [header, '']
    data = [(k, v) for k, v in d.items()]
    print(tabulate(data, headers=headers))


class BaseQuery:
    def __init__(self, index):
        self.index = index
        self.df = False
        if self.index is None:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        else:
            self.model = self.index.model

    def score(self, target_dnn_output):
        raise NotImplementedError

    def finish_index_building(self):
        ### only perform this operation if the index is of type seiden
        if 'EKO' in repr(self.index):
            print('[Here exe EKO MABS]')
            index = self.index
            target_dnn = self.index.target_dnn_cache
            scoring_func = self.score
            index.build_additional_anchors(target_dnn, scoring_func)
        else:
            print('[Here exe tasti-Pi]')

    # 标签传播
    def propagate(self, target_dnn_cache, reps, topk_reps, topk_distances, images=None, index_cache=None):
        if not self.df:
            score_fn = self.score
            print("[propagate_images]: ", len(images))
            print("[top_reps]: ", topk_reps)
            y_true = np.array(
                [tasti.DNNOutputCacheFloat(target_dnn_cache, score_fn, idx, self.model, images, self.index) for idx in
                 range(len(topk_reps))]
            )
            print("[propa_y_true]: ", y_true)
            y_pred = np.zeros(len(topk_reps))
            if self.index.config.label_propagate_al == 1:
                # if 'EKO_sigmoid' in repr(self.index):
                print("[exec_sigmoid]")
                ### custom building of label propagation....
                ### so we now have the score, we just need to generate the y_pred values based on topk_reps and topk_distances
                for i in tqdm(range(len(y_pred)), 'Sigmoid based Propagation'):
                    weights = topk_distances[i]  ### we know there are only 2 distances...
                    reps = topk_reps[i]  ### we know there is only two reps
                    counts = y_true[reps]

                    left, right = float(counts[0]), float(counts[1])
                    ### compute the values
                    x_mid = (reps[0] + reps[1]) // 2
                    amp = abs(left - right)
                    y_low = min(left, right)

                    if left <= right:
                        y_pred[i] = amp / (1 + np.exp(-(i - x_mid))) + y_low
                    else:
                        y_pred[i] = amp / (1 + np.exp((i - x_mid))) + y_low
            else:
                print("[Propagate]: ", topk_distances)
                for i in tqdm(range(len(y_pred)), 'Propagation ： Here exe linear propagate'):
                    weights = topk_distances[i]
                    weights = np.sum(weights) - weights
                    weights = weights / weights.sum()
                    # print('[topk_distances]: ', topk_distances[i])
                    # print('[weights]: ', weights)
                    counts = y_true[topk_reps[i]]
                    # print('[counts]: ', counts)
                    y_pred[i] = np.sum(counts * weights)
                print('[Here has completed propagate]')
        else:
            y_true = self.score(target_dnn_cache.df)
            y_pred = np.zeros(len(topk_reps))
            weights = topk_distances
            weights = np.sum(weights, axis=1).reshape(-1, 1) - weights
            weights = weights / weights.sum(axis=1).reshape(-1, 1)
            counts = np.take(y_true, topk_reps)
            y_pred = np.sum(counts * weights, axis=1)
        # print("pred Label",y_pred)
        # print("true Label",y_true)
        return y_pred, y_true

    def execute(self):
        raise NotImplementedError


class AggregateQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def _execute(self, err_tol=0.01, confidence=0.05, y=None, images=None):
        propagate_time = 0
        if y == None:
            self.finish_index_building()
            st = time.perf_counter()
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists,
                images=images,
                index_cache=self.index.index_cache
            )
            et = time.perf_counter()
            propagate_time = et - st
            print('[Here is propagate time]:', propagate_time)
        else:
            y_pred, y_true = y
        print('[Here has completed the MABS && label propagate]')
        #### here we will save the array...
        self.y_pred = y_pred
        self.y_true = y_true
        # print('[y_pred]: ', self.y_pred)
        # print('[y_true]: ', self.y_true)
        r = max(1, np.amax(np.rint(y_pred)))
        # print('[Here is index_cahce_len 1]: ', len(self.index.index_cache))
        # print("r---here", r)
        st = time.perf_counter()
        sampler = ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r)
        estimate, nb_samples = sampler.sample()
        et = time.perf_counter()
        Blazit_time = et - st
        print('[Here is Blazit time]:', Blazit_time)
        # print('[Here is index_cahce_len 2]: ', len(self.index.index_cache))
        res = {
            'initial_estimate': y_pred.sum(),
            'blazeit_estimate': estimate,
            'nb_samples': nb_samples,
            'y_pred': y_pred,
            'y_true': y_true,
            'query_time': Blazit_time + propagate_time,
            'Blazit time': Blazit_time,
            'propagate_time': propagate_time
        }
        print("[estimate]: ", res['blazeit_estimate'])
        print('[initial_estimate]: ', res['initial_estimate'])

        return res

    def execute(self, err_tol=0.01, confidence=0.05, y=None):
        res = self._execute(err_tol, confidence, y)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, err_tol=0.01, confidence=0.05, images=None, y=None, save_dir=None):
        print(f'[error_tol]: {err_tol} , confidence: {confidence}')
        res = self._execute(err_tol, confidence, y, images)
        # stp = time.perf_counter()
        # res['actual_estimate'] = res['y_true'].sum()  # expensive
        # etp = time.perf_counter()
        # print('[y_sum_time]: ', etp - stp)
        # print('[actual_estimate]', res['actual_estimate'])
        # print_dict(res, header=self.__class__.__name__)
        return res

    def get_results(self, err_tol=0.01, confidence=0.05, y=None, save_dir=None):
        res = self._execute(err_tol, confidence, y)
        res['actual_estimate'] = res['y_true'].sum()  # expensive
        result = f"nb_samples: {res['nb_samples']}"
        # print(result)
        return result


class LimitQuery(BaseQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)

    def execute(self, want_to_find=5, nb_to_find=10, GAP=300, y=None):
        if y == None:
            self.finish_index_building()
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        order = np.argsort(y_pred)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        for ind in order:
            if ind in visited:
                continue
            nb_calls += 1
            if float(y_true[ind]) >= want_to_find:
                ret_inds.append(ind)
                for offset in range(-GAP, GAP + 1):
                    visited.add(offset + ind)
            if len(ret_inds) >= nb_to_find:
                break
        res = {
            'nb_calls': nb_calls,
            'ret_inds': ret_inds
        }
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, want_to_find=5, nb_to_find=10, GAP=300, y=None):
        return self.execute(want_to_find, nb_to_find, GAP, y)


class SUPGPrecisionQuery(BaseQuery):

    def score(self, target_dnn_output):
        raise NotImplementedError

    def _execute(self, budget, y=None, images=None):
        if y == None:
            # 2.1 MABS
            self.finish_index_building()
            # 2.2 标签传播
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists,
                images=images,
                index_cache=self.index.index_cache
            )
        else:
            y_pred, y_true = y
        self.y_pred = y_pred
        self.y_true = y_true
        print('y_pre', self.y_pred)
        # print('[Here is index_cahce_len 1]: ', len(self.index.index_cache))
        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        print(fr'[budget: {budget} , index_cache_len: {len(self.index.index_cache)}]')
        if budget < len(self.index.index_cache):
            budget = len(self.index.index_cache)
            print('[here start self_adapt budget]')
        query = ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=0.95, delta=0.1,
            budget=budget
        )
        selector = ImportancePrecisionTwoStageSelector(query, source, sampler)
        inds = selector.select()
        # print('[Here is index_cahce_len 2]: ', len(self.index.index_cache))

        # print('[anwser_inds]: ', inds)
        print('[len_inds]: ', len(inds))
        # for ind in inds:
        #     print(ind)
        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            # 'y_true': y_true,
            'y_pred': y_pred,
            # 'source': source
        }
        return res

    def execute(self, budget, y=None):
        res = self._execute(budget, y)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, budget, y=None, images=None):
        res = self._execute(budget, y, images)
        # source = res['source']
        inds = res['inds']
        print('len_inds: ', len(inds))
        # nb_got = np.sum(source.lookup(inds))
        # nb_true = res['y_true'].sum()
        # precision = nb_got / len(inds)
        # recall = nb_got / nb_true
        # res['precision'] = precision
        # res['recall'] = recall
        # # res['precision'] = 'test'
        # # res['recall'] = 'test'
        # print_dict(res, header=self.__class__.__name__)
        return res

    def get_results(self, budget, y=None):
        res = self._execute(budget, y)
        source = res['source']
        inds = res['inds']
        nb_got = np.sum(source.lookup(inds))
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        result = f'Precision: {precision}, Recall: {recall}'
        return result


class SUPGRecallQuery(SUPGPrecisionQuery):
    def _execute(self, budget, y=None, images=None):
        if y == None:
            # MABS采样
            self.finish_index_building()
            # 标签传播
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists,
                images=images,
                index_cache=self.index.index_cache
            )
        else:
            y_pred, y_true = y

        self.y_pred = y_pred
        self.y_true = y_true
        # for pre in self.y_pred:
        #     print('[y_pre]: ',pre)
        # for yp ,yt in zip(self.y_pred,y_true):
        #     print(yp,'   ',yt*1 )
        # print(fr'[budget: {budget} , index_cache_len: {len(self.index.index_cache)}]')
        # if budget < int(len(self.index.index_cache)*0.5):
        #     budget = int(len(self.index.index_cache)*0.5)
        #     print('[here start self_adapt budget]')
        recall_config = self.index.config
        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()

        query = ApproxQuery(
            qtype='rt',
            min_recall=recall_config.user_recall_threshold, min_precision=recall_config.user_precision_threshold,
            delta=recall_config.user_confidence_threshold,
            budget=budget
        )
        print('[budget]:,', budget)
        selector = RecallSelector(query, source, sampler, sample_mode='sqrt')
        inds = selector.select()

        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source
        }
        return res

    def execute(self, budget, y=None):
        res = self._execute(budget, y)
        # print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, budget, y=None, images=None):
        # 这行代码执行MABS
        print('[budget]: ', budget)
        res = self._execute(budget, y, images=images)
        source = res['source']
        inds = res['inds']
        # for ind in inds:
        #     print('[inds]: ',ind)
        print('len(inds)=', len(inds))
        # nb_got = np.sum(source.lookup(inds))
        # nb_true = res['y_true'].sum()
        # precision = nb_got / len(inds)
        # recall = nb_got / nb_true
        # res['precision'] = precision
        # res['recall'] = recall
        # print_dict(res, header=self.__class__.__name__)
        return res
