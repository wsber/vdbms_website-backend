"""
This file is the newest version of EKO --
We will implement three different optimizations

1. I-frames
    Instead of using constant rate sampling, we will retrieve the I-frame information
    If # of bduget is greater than the number of I-frames, take entire i-frame set
    Select the rest using exploration / exploitation during query execution

    else:
        randomly select subset of i-frames

2. Label Propagation -- sigmoid function (aka. smoothing function)


3. MAB -- this replaces the alternating exploration and exploitation method we currently have.
"""
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

"""
cluster_dict 是用于存储每个时间片段（cluster）的信息的字典。它的每个元素表示一个时间片段，其中键是时间片段的起始和结束索引对，值是包含该时间片段内帧的索引、帧的得分以及时间片段的距离（奖励）的字典。
具体来说，cluster_dict 的每个元素具有以下结构：
键：时间片段的起始和结束索引对 (start_idx, end_idx)。
值：一个字典，包含以下键值对：
'members'：包含该时间片段内帧的索引的列表。
'cache'：包含该时间片段内帧得分的列表。
'distance'：表示该时间片段的距离（奖励），通常是帧得分的方差。
通过这种方式，cluster_dict 记录了每个时间片段的信息，用于多臂赌博机（MAB）算法中的选择和更新过程。
"""

import csv
import os
import torch
import numpy as np
import torchvision
import sys
import pandas as pd

sys.path.append('/nethome/jbang36/seiden')
from udfs.yolov5.utils.general import non_max_suppression
from sklearn.linear_model import LinearRegression
from benchmarks.stanford.tasti.tasti.index import Index
from benchmarks.stanford.tasti.tasti.config import IndexConfig
from benchmarks.stanford.tasti.tasti.seiden.data.data_loader import ImageDataset, LabelDataset
from tqdm import tqdm
import time
from src.motivation.tasti_wrapper import InferenceDataset
from src.system_architecture.alternate import EKO_alternate
from src.iframes.pyav_utils import PYAV_wrapper
from collections import OrderedDict
import random


class EKO_mab(EKO_alternate):
    def __init__(self, config, images, video_f, anchor_percentage=0.8, c_param=2, keep=False):
        self.images = images
        self.pyav = PYAV_wrapper(video_f)  ## need up to .mp4
        self.c_param = c_param
        self.anchor_percentage = anchor_percentage
        self.keep = keep
        self.oracle_count = 0
        self.transform = self.inference_transforms()
        self.recursion_deep = 0
        self.reps_hash = set()
        super().__init__(config, images)

    def __repr__(self):
        return 'EKO'

    def do_bucketting(self, percent_fpf=0.75):
        if self.config.do_bucketting:
            # 1.1抽完了所有的代表帧
            self.reps, self.topk_reps, self.topk_dists = self.calculate_rep_methodology()
            np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
            np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
            np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)

        else:
            print(os.path.join(self.cache_dir, 'reps.npy'))
            self.reps = np.load(os.path.join(self.cache_dir, 'reps.npy'))
            self.topk_reps = np.load(os.path.join(self.cache_dir, 'topk_reps.npy'))
            self.topk_dists = np.load(os.path.join(self.cache_dir, 'topk_dists.npy'))

    def calculate_rep_methodology(self):
        rep_indices, dataset_length = self.get_reps()  ### we need to sort the reps
        # print(rep_indices, dataset_length)
        top_reps = self.calculate_top_reps(dataset_length, rep_indices)
        top_dists = self.calculate_top_dists(dataset_length, rep_indices, top_reps)
        return rep_indices, top_reps, top_dists

    #  get_reps 的作用是从视频中选择关键帧（I帧）作为代表帧集合。具体步骤如下：
    def get_reps(self):
        """
        We will select the i-frames and that's it
        :param dataset_length:
        :return:
        """
        self.model.eval()
        dataset_length = len(self.images)
        # 使用self.pyav.get_iframe_indices()函数获取视频中所有关键帧（I帧）的索引。
        iframe_indices = self.pyav.get_iframe_indices_folder()
        # iframe_indices = self.pyav.get_iframe_indices()
        n_reps = self.config.nb_buckets
        print("[here is Iframes Number]: ", len(iframe_indices))
        print("[here is self.config.nb_buckets]: ", n_reps)
        # print("[iframe_indices]: ", iframe_indices)
        print('[N , alpha , alpha*N]： ', n_reps, self.anchor_percentage, int(n_reps * self.anchor_percentage))
        index_construction_reps = int(n_reps * self.anchor_percentage)

        print('[total number of iframes]: ', len(iframe_indices))
        print('[total number of anchors selected in index construction]: ', index_construction_reps)

        ### but we always have to include the first and last indices of the dataset
        if n_reps < 2:
            raise ValueError('Number of Reps too Low')

        rep_indices = [0, dataset_length - 1]
        ### If it ends up happening that total number of i-frames is less than index_construction_reps, then we just sample them in the latter phase.
        if index_construction_reps >= len(iframe_indices) + 2:
            print("[exe -- if index_construction_reps >= len(iframe_indices) + 2]: ")
            print('[iframe_indices]: ', iframe_indices)
            print(dataset_length)
            rep_indices.extend(iframe_indices)
            self.reps_hash.update(rep_indices)
            rep_indices = self.get_random_reps_iframes(rep_indices, index_construction_reps - len(iframe_indices),  dataset_length)
            rep_indices = list(set(rep_indices))
            rep_indices = sorted(rep_indices)
        else:
            # 下面这一行代码有问题
            print('[here index_consturction]')
            subset = list(np.random.choice(iframe_indices, index_construction_reps, replace=False))
            rep_indices.extend(subset)
            rep_indices = list(set(rep_indices))
            rep_indices = rep_indices[:index_construction_reps]
            rep_indices = sorted(rep_indices)

        print('[final number that has been selected]: ', len(rep_indices))
        print('[rep_indices]: ', rep_indices)
        self.base_reps = rep_indices
        return rep_indices, dataset_length

    def get_random_reps_iframes(self, rep_indices, iter, dataset_length):
        ct = 0
        while ct < iter - 2:
            selected_rep = random.randint(0, dataset_length - 1)
            if selected_rep not in self.reps_hash:
                rep_indices.append(selected_rep)
                self.reps_hash.add(selected_rep)
                ct += 1
        return rep_indices

    # calculate_top_reps函数用于根据给定的代表帧索引列表
    # rep_indices来生成代表帧集合。它通过在视频帧序列中将每个代表帧索引范围内的帧都标记为该代表帧的起始和结束索引来实现。
    # 最后，生成一个包含所有帧的起始和结束索引对的数组top_reps并返回。
    def calculate_top_reps(self, dataset_length, rep_indices):
        """
        Choose representative frames based on systematic sampling

        :param dataset_length:
        :param rep_indices:
        :return:
        """
        top_reps = np.ndarray(shape=(dataset_length, 2), dtype=np.int32)

        ### do things before the first rep
        for i in range(len(rep_indices) - 1):
            start = rep_indices[i]
            end = rep_indices[i + 1]
            top_reps[start:end, 0] = start
            top_reps[start:end, 1] = end

        ### there could be some left over at the end....
        last_rep_indices = rep_indices[-1]
        top_reps[last_rep_indices:, 0] = rep_indices[-2]
        top_reps[last_rep_indices:, 1] = rep_indices[-1]
        # print("[top_reps]: ", top_reps)
        return top_reps

    # 这个函数calculate_top_dists 的作用是计算当前帧与最近的代表帧之间的时间距离。在这个函数中，
    # dataset_length是数据集的长度，
    # rep_indices是代表帧的索引，
    # top_reps是代表帧的起始和结束索引对。这个函数返回一个数组，其中每一行包含了当前帧与其最近的代表帧的时间距离。
    # 具体来说，这个函数做了以下几件事情：
    # 1.创建了一个数组top_dists，其形状为(dataset_length, 2)，数据类型为 np.int32，用于存储每一帧与其最近的代表帧的时间距离。
    # 2.对于数据集中的每一帧，计算其与其最近的代表帧的时间距离，并将结果存储在 top_dists数组中。
    def calculate_top_dists(self, dataset_length, rep_indices, top_reps):
        """
        Calculate distance based on temporal distance between current frame and closest representative frame
        :param dataset_length:
        :param rep_indices:
        :param top_reps:
        :return:
        """
        print('[calculate_top_dists_dataset_length]: ', dataset_length)
        ### now we calculate dists
        top_dists = np.ndarray(shape=(dataset_length, 2), dtype=np.int32)

        for i in range(dataset_length):
            # top_dists[i, 0] = abs(i - top_reps[i, 1])
            # top_dists[i, 1] = abs(i - top_reps[i, 0])
            top_dists[i, 0] = abs(i - top_reps[i, 0])
            top_dists[i, 1] = abs(i - top_reps[i, 1])
        # print("[top_dists]: ", top_dists)
        # print('[len_top_dists]: ', len(top_dists))
        return top_dists

    def _get_closest_reps(self, rep_indices, curr_idx):
        result = []

        for i, rep_idx in enumerate(rep_indices):
            if rep_idx - curr_idx >= 0:
                result.append(rep_indices[i - 1])
                result.append(rep_indices[i])
                break
        return result

    # 这个函数执行MABS采样
    def build_additional_anchors(self, target_dnn_cache, scoring_func):
        """
        We will replace this function with the mab implementation...
        Steps are as follows:
        1. We create the clusters....and keep them fixed
        2. We will randomly draw from the cluster
        3. We will
        :param target_dnn_cache:
        :param scoring_func:
        :return:
        """
        n_reps = self.config.nb_buckets
        ### we need to keep track of cluster boundaries and rep_indices
        rep_indices = self.base_reps
        rep_indices = sorted(rep_indices)
        curr_len = len(rep_indices)
        topk_reps = self.topk_reps
        # print("reps_numbers", curr_len)
        # for i in rep_indices:
        #     print(i)

        ### init distances / clusters
        ### for given distances, we iterate by selecting the argmax
        ### top reps, top dists are calculated in the same manner as before.
        length = rep_indices[-1] - rep_indices[0] + 1
        print("[rep_indices] :", len(rep_indices))
        cluster_dict = self.init_label_distances(rep_indices, target_dnn_cache, scoring_func, length)
        # 迭代选择新的代表帧，直到达到指定的代表帧数量。
        all_num = n_reps - curr_len
        update_step_length = float((all_num)) * 0.025
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "algorithm_type_group",
            {
                "type": "algorithm.update",
                "algorithm_type": 'mab_sampling',
                "consumer_type": 'algorithm_type'
            }
        )
        for i in tqdm(range((n_reps - curr_len)), desc='MABSampling: Here exe select N-alpha*N times MABSampling'):
            # 通过多臂赌博机（MAB）进行迭代选择新的代表帧：在每次迭代中，调用 select_rep 函数从簇中选择新的代表帧，并更新簇的信息。选择代表帧的方式通过多臂赌博机算法进行，
            # 具体的选择依赖于每个簇的奖励和探索因子。随着迭代的进行，新的代表帧会被加入到 rep_indices 中，并更新 cluster_dict。
            new_rep, cluster_key = self.select_rep(cluster_dict, rep_indices)
            rep_indices.append(new_rep)
            cluster_dict = self.update(cluster_dict, cluster_key, new_rep, target_dnn_cache, scoring_func)
            if i % 10 == 0 or i == all_num -1:
                progress = (i + 1) / all_num * 100
                # print('[progress]: ', progress)
                async_to_sync(channel_layer.group_send)(
                    "progress_group",
                    {
                        "type": "progress.update",
                        "progress": progress,
                        "algorithm_type": 'mab_sampling'
                    }
                )
        rep_indices = sorted(rep_indices)
        assert (len(rep_indices) == len(set(rep_indices)))

        # # 使用历史缓存的视频结构化数据
        history_structured_reps = self.get_history_structured_data_frameId()
        print('[here history structured data rep_indices]: ', history_structured_reps)
        history_structured_reps.extend(rep_indices)
        rep_indices = sorted(list(set(history_structured_reps)))
        print('[here mix history structured data rep_indices with new rep_indeces]: ', rep_indices)
        print('[build_rep_indices]: ', rep_indices)
        print('[len_rep_indices]: ', len(rep_indices))
        dataset_length = len(topk_reps)
        top_reps = self.calculate_top_reps(dataset_length, rep_indices)
        top_dists = self.calculate_top_dists(dataset_length, rep_indices, top_reps)
        self.cluster_dict = cluster_dict
        self.reps = rep_indices
        self.topk_reps = top_reps
        self.topk_dists = top_dists

        # print("[top_reps_num]: ", len(topk_reps), "\n", topk_reps)

        np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
        np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
        np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)
        # 下面是测试中间结果输出代码
        # for cluster_key, cluster_info in cluster_dict.items():
        #     start_idx, end_idx = cluster_key
        #     members = cluster_info['members']
        #     cache = cluster_info['cache']
        #     distance = cluster_info['distance']
        #
        #     print(f"Cluster: ({start_idx}, {end_idx})")
        #     print(f"Members: {members}")
        #     print(f"Cache: {cache}")
        #     print(f"Distance: {distance}")
        #     print()

        if self.keep:
            self.base_reps = rep_indices

    def get_history_structured_data_frameId(self):
        # 指定要读取的文件路径
        print('[history_video_name]: ', self.config.video_name)
        output_dir = fr"/home/wangshuo_20/pythonpr/VDBMS_ws/media/videoCacheData/{self.config.video_name}/seiden"
        # output_dir = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden"
        # output_dir = fr"D:/Projects/PyhtonProjects/thesis/tasti_data/cache/{self.config.video_name}/seiden"
        filename = f"{self.config.video_name}_reps_frame.txt"
        frame_file_path = os.path.join(output_dir, filename)
        # 创建一个空列表用于存储读取的帧号数据
        history_reps = []
        # 打开文件并读取数据
        try:
            with open(frame_file_path, 'r') as framefile:
                # 逐行读取文件内容
                for line in framefile:
                    # 去除行末的换行符并将内容转换为整数类型，然后添加到 history_reps 列表中
                    frame_number = int(line.strip())
                    history_reps.append(frame_number)
            # 打印读取成功的消息和历史帧号数据
            print(f"成功从文件 '{frame_file_path}' 中读取帧号数据到 history_reps 列表。")
            print("历史帧号数据：", history_reps)
        except FileNotFoundError:
            # 处理文件不存在的情况
            print(f"文件 '{frame_file_path}' 不存在。返回空列表。")
            history_reps = []
        return history_reps

    # 每次采样后更新每个臂的奖励
    def update(self, cluster_dict, cluster_key, rep_idx, target_dnn_cache, scoring_func):
        ### I mean we can just update everything here.....
        start_idx, end_idx = cluster_key
        results = self.run_oracle_model(rep_idx)
        # max_confidence = self.get_confidence_of_yolvo5s_result(results)
        # print(results.pandas().xyxy[0])
        cluster_dict[(start_idx, end_idx)]['members'].append(rep_idx)
        # 这个target_dnn_cache[rep_idx]是当前视频帧在神经网络检测后对是否符合'car'的得分，scoring_func（）用于判断当前帧reward的取值
        cluster_dict[(start_idx, end_idx)]['cache'].append(scoring_func(results))
        # print("score:   ", scoring_func(target_dnn_cache[rep_idx]),"    target_dnn_cache[rep_idx]:   ",target_dnn_cache[rep_idx])
        # print('[update_mean]')
        # cluster_dict[(start_idx, end_idx)]['distance'] = np.var(cluster_dict[(start_idx, end_idx)]['cache'])
        cluster_dict[(start_idx, end_idx)]['distance'] = np.mean(cluster_dict[(start_idx, end_idx)]['cache'])
        cluster_dict[(start_idx, end_idx)]['hash_members'].add(rep_idx)
        return cluster_dict

    def select_rep_alpha(self, cluster_dict, rep_indices, alpha=0.1):
        if random.random() <= alpha:
            # print('[Random]')
            return self.random_select_rep(cluster_dict)
        else:
            # print('[MABS Selection]')
            return self.select_rep(cluster_dict, rep_indices)

    # MAB采样一次，从各个片段中选择一个最优片段，从该片段随机抽取一帧
    def select_rep(self, cluster_dict, rep_indices):
        ### compute the distances of each cluster and select based on the formula
        mab_values = []
        c_param = self.c_param
        for cluster_key in cluster_dict:
            reward = cluster_dict[cluster_key]['distance']
            members = cluster_dict[cluster_key]['members']
            mab_value = reward + c_param * np.sqrt(2 * np.log(len(rep_indices)) / len(members))
            # print(fr'[reward:{reward} , c*b: {c_param * np.sqrt(2 * np.log(len(rep_indices)) / len(members))}]')
            mab_values.append(mab_value)
        assert (len(mab_values) == len(cluster_dict.keys()))
        # print(mab_values)

        selected_cluster = np.argmax(mab_values)
        # print('selected cluster: ', list(cluster_dict.keys())[selected_cluster], 'mab value is: ', mab_values[selected_cluster])

        start_idx, end_idx = list(cluster_dict.keys())[selected_cluster]

        choices = []
        for i in range(start_idx, end_idx + 1):
            if i not in cluster_dict[(start_idx, end_idx)]['hash_members']:
                choices.append(i)
        # 具体来说，如果在当前片段中找不到未被选择的帧，即 len(choices) == 0，则需要采取措施来处理这种情况。在这段代码中，
        # 处理方式是将当前片段的距离（奖励）设置为0，然后通过递归调用 select_rep 函数从其他片段中选择新的代表帧。
        # 递归调用的参数保持不变，仍然传递了当前的 cluster_dict 和 rep_indices。
        if len(choices) == 0:
            print('{run in recursion}')
            self.recursion_deep += 1
            print(fr'recursion_deep: {self.recursion_deep}')
            cluster_dict[(start_idx, end_idx)]['distance'] = -1
            rep_idx, (start_idx, end_idx) = self.select_rep(cluster_dict, rep_indices)
        else:
            rep_idx = np.random.choice(choices, 1)[0]

        # print('selected cluster: ', selected_cluster, 'selected ', rep_idx, 'max mab value: ', np.max(mab_values))
        return rep_idx, (start_idx, end_idx)

    def select_rep_iter(self, cluster_dict, rep_indices):
        ### compute the distances of each cluster and select based on the formula
        N = len(cluster_dict)
        mab_values = []
        c_param = self.c_param
        for cluster_key in cluster_dict:
            reward = cluster_dict[cluster_key]['distance']
            members = cluster_dict[cluster_key]['members']
            mab_value = reward + c_param * np.sqrt(2 * np.log(len(rep_indices)) / len(members))
            mab_values.append(mab_value)
        for it in range(N):
            assert (len(mab_values) == len(cluster_dict.keys()))
            # print(mab_values)
            selected_cluster = np.argmax(mab_values)
            # print('selected cluster: ', list(cluster_dict.keys())[selected_cluster], 'mab value is: ', mab_values[selected_cluster])
            start_idx, end_idx = list(cluster_dict.keys())[selected_cluster]
            choices = []
            for i in range(start_idx, end_idx + 1):
                if i not in cluster_dict[(start_idx, end_idx)]['members']:
                    choices.append(i)

            # 具体来说，如果在当前片段中找不到未被选择的帧，即 len(choices) == 0，则需要采取措施来处理这种情况。在这段代码中，
            # 处理方式是将当前片段的距离（奖励）设置为0，然后通过递归调用 select_rep 函数从其他片段中选择新的代表帧。
            # 递归调用的参数保持不变，仍然传递了当前的 cluster_dict 和 rep_indices。
            if len(choices) == 0:
                cluster_dict[(start_idx, end_idx)]['distance'] = 0
                rep_idx, (start_idx, end_idx) = self.select_rep(cluster_dict, rep_indices)
            else:
                rep_idx = np.random.choice(choices, 1)[0]

        # print('selected cluster: ', selected_cluster, 'selected ', rep_idx, 'max mab value: ', np.max(mab_values))
        return rep_idx, (start_idx, end_idx)

    def random_select_rep(self, cluster_dict):
        N = len(cluster_dict)
        for it in range(N):
            # 随机生成 0 到 n 之间的整数
            selected_cluster = random.randint(0, N - 1)
            start_idx, end_idx = list(cluster_dict.keys())[selected_cluster]
            choices = []
            for i in range(start_idx, end_idx + 1):
                if i not in cluster_dict[(start_idx, end_idx)]['members']:
                    choices.append(i)
            if len(choices) == 0:
                pass
            else:
                rep_idx = np.random.choice(choices, 1)[0]
        return rep_idx, (start_idx, end_idx)

    def init_label_distances(self, cluster_boundaries, target_dnn_cache, scoring_func, length):
        #### we will generate the cluster_dictionary
        cluster_dict = OrderedDict()
        #### instead of making cluster dict based on cluster boundaries, we have to get the total number of frames and create boundaries based on that
        N = 100
        step_size = length // N
        ####
        rep_idx = 0
        for i in range(0, length, step_size):
            start, end = i, min(i + step_size - 1, length - 1)
            #### find all rep_indices that fall within this boundary
            corresponding_reps = []
            while rep_idx < len(cluster_boundaries):
                if cluster_boundaries[rep_idx] <= end:
                    corresponding_reps.append(cluster_boundaries[rep_idx])
                else:
                    break
                rep_idx += 1
            cache = []
            # 这个地方需要优化，得用到cache缓存以下
            for rep in corresponding_reps:
                results = self.run_oracle_model(rep)
                cache.append(scoring_func(results))

            cluster_dict[(start, end)] = {
                'members': corresponding_reps,
                'cache': cache,
                'distance': np.var(cache),
                'hash_members': set(corresponding_reps)
            }
            # print("inited_cluster_dict: ", cluster_dict)
        print('[len_of_cluster_dict]: ', len(cluster_dict))
        return cluster_dict

    def run_oracle_model_(self, rep):
        # print('[Here is seiden_run]')
        if rep in self.index_cache:
            return self.index_cache[rep]
        else:
            with torch.no_grad():
                tmp_image = self.transform(self.images[rep])
                tmp_image = tmp_image.unsqueeze(0)
                result = self.model(tmp_image)
                tt = result.clone()
                y = non_max_suppression(tt, conf_thres=0.01, iou_thres=0.45, classes=None)  # NMS
                detections = y[0]
                if len(detections) > 0:
                    detections = detections.to('cpu')
                    boxes = detections[:, :4]
                    scores = detections[:, 4]
                    labels = detections[:, 5].int()
                    for box, score, label in zip(boxes, scores, labels):
                        if self.classes[label] == 'car':
                            inner_dict = {
                                'object_name': self.classes[label],
                                'confidence': float(score),  # Convert tensor to float
                                'xmin': int(box[0]),
                                'ymin': int(box[1]),
                                'xmax': int(box[2]),
                                'ymax': int(box[3])
                            }
                            if rep in self.index_cache:
                                self.index_cache[rep].append(inner_dict)
                            else:
                                self.index_cache[rep] = [inner_dict]
                        if rep not in self.index_cache:
                            self.index_cache[rep] = [{
                                'object_name': None,
                                'confidence': -1,  # Convert tensor to float
                                'xmin': 0,
                                'ymin': 0,
                                'xmax': 0,
                                'ymax': 0
                            }]
                        return self.index_cache[rep]
        # print(self.index_cache)

    def run_oracle_model(self, rep):
        # print('[run_seiden_oracle]')
        if rep in self.index_cache:
            # print('[cache_hit]')
            return self.index_cache[rep]
        else:
            with torch.no_grad():
                if rep >= len(self.images):
                    rep = len(self.images) - 1
                result = self.model(self.images[rep])
                # result = result.to('cpu')
                # Iterate through detections for each image in the batch
                detections = result.xyxy[0]
                if len(detections) > 0:
                    # Iterate through detections for this image
                    for detection in detections:
                        # Extract bounding box coordinates and confidence score
                        bbox = detection[:4]  # First four elements are usually bbox coordinates
                        confidence = detection[4]  # Fifth element is often the confidence score
                        object_name = self.model.names[int(detection[5])]  # Get class name using model.names
                        if (object_name == 'car'):
                            # Create the inner dictionary
                            inner_dict = {
                                'object_name': object_name,
                                'confidence': confidence.item(),  # Convert tensor to float
                                'xmin': int(bbox[0]),
                                'ymin': int(bbox[1]),
                                'xmax': int(bbox[2]),
                                'ymax': int(bbox[3])
                            }
                            # print('[inner_dict]: ', inner_dict)
                            # Add the inner dictionary to your index_cache
                            if rep in self.index_cache:
                                self.index_cache[rep].append(inner_dict)
                            else:
                                self.index_cache[rep] = [inner_dict]
                    if rep not in self.index_cache:
                        self.index_cache[rep] = [{
                            'object_name': None,
                            'confidence': -1,  # Convert tensor to float
                            'xmin': 0,
                            'ymin': 0,
                            'xmax': 0,
                            'ymax': 0
                        }]
                    # print(self.index_cache)
                    return self.index_cache[rep]
                else:
                    self.index_cache[rep] = [{
                        'object_name': None,
                        'confidence': -1,  # Convert tensor to float
                        'xmin': 0,
                        'ymin': 0,
                        'xmax': 0,
                        'ymax': 0
                    }]
                    return self.index_cache[rep]

    # images = []
    # for i in rep :
    #     images.append(self.images[i])
    # images = np.array(images)

    def run_oracle_model_full(self, rep):
        # print('[run_full]')
        if rep in self.full_index_cache:
            # print('[cache_hit]')
            return self.full_index_cache[rep]
        else:
            with torch.no_grad():
                result = self.model(self.images[rep])
                # result = result.to('cpu')
                # Iterate through detections for each image in the batch
                detections = result.xyxy[0]
                if len(detections) > 0:
                    # Iterate through detections for this image
                    for detection in detections:
                        # Extract bounding box coordinates and confidence score
                        bbox = detection[:4]  # First four elements are usually bbox coordinates
                        confidence = detection[4]  # Fifth element is often the confidence score
                        object_name = self.model.names[int(detection[5])]  # Get class name using model.names
                        if (object_name == 'car'):
                            # Create the inner dictionary
                            inner_dict = {
                                'object_name': object_name,
                                'confidence': confidence.item(),  # Convert tensor to float
                                'xmin': int(bbox[0]),
                                'ymin': int(bbox[1]),
                                'xmax': int(bbox[2]),
                                'ymax': int(bbox[3])
                            }
                            # print('[inner_dict]: ', inner_dict)
                            # Add the inner dictionary to your index_cache
                            if rep in self.full_index_cache:
                                self.full_index_cache[rep].append(inner_dict)
                            else:
                                self.full_index_cache[rep] = [inner_dict]
                    if rep not in self.full_index_cache:
                        self.full_index_cache[rep] = [{
                            'object_name': None,
                            'confidence': -1,  # Convert tensor to float
                            'xmin': 0,
                            'ymin': 0,
                            'xmax': 0,
                            'ymax': 0
                        }]
                    # print(self.full_index_cache)
                    return self.full_index_cache[rep]
                else:
                    self.index_cache[rep] = [{
                        'object_name': None,
                        'confidence': -1,  # Convert tensor to float
                        'xmin': 0,
                        'ymin': 0,
                        'xmax': 0,
                        'ymax': 0
                    }]
                    return self.index_cache[rep]

    def save_index_cache(self):
        """
        Writes a CSV file from the given index dictionary.
        Args:
            index (dict): The index dictionary with numerical keys and inner dictionaries as values.
            output_dir (str): The directory to save the CSV file.
            filename (str): The name of the CSV file.
        """
        output_dir = fr"/home/wangshuo_20/pythonpr/VDBMS_ws/media/videoCacheData/{self.config.video_name}/seiden"
        # output_dir = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden"
        # output_dir = fr"D:/Projects/PyhtonProjects/thesis/tasti_data/cache/{self.config.video_name}/seiden"
        filename = fr"{self.config.video_name}_cache.csv"
        index = self.index_cache
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Construct the full file path
        filepath = os.path.join(output_dir, filename)
        # Open the CSV file for writing
        # print("[write csv index_cache]",index)
        with open(filepath, 'w', newline='') as csvfile:
            # Define the column headers
            fieldnames = ['frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'ind']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write the header row
            writer.writeheader()
            # Iterate through the index dictionary and write rows
            for key, inner_dicts in index.items():
                # Create a dictionary with the required structure
                for inner_dict in inner_dicts:
                    row_dict = {
                        'frame': key,
                        'object_name': inner_dict.get('object_name'),
                        'confidence': inner_dict.get('confidence'),
                        'xmin': inner_dict.get('xmin'),
                        'ymin': inner_dict.get('ymin'),
                        'xmax': inner_dict.get('xmax'),
                        'ymax': inner_dict.get('ymax'),
                        'ind': key  # Use the key as the 'ind' value
                    }
                    # Write the row to the CSV file
                    writer.writerow(row_dict)
        # 下面对缓存帧保存，用于后续对该视频的重复查询--这里先简化到对同一查询语句的复用
        # 获取排序后的帧号列表（整数类型）
        sorted_keys = sorted(index.keys())
        print('[len_cache_frame]: ', len(sorted_keys))
        content_to_write = '\n'.join(str(key) for key in sorted_keys)
        output_dir = fr"/home/wangshuo_20/pythonpr/VDBMS_ws/media/videoCacheData/{self.config.video_name}/seiden"
        # output_dir = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden"
        # output_dir = fr"D:/Projects/PyhtonProjects/thesis/tasti_data/cache/{self.config.video_name}/seiden"
        filename = fr"{self.config.video_name}_reps_frame.txt"
        frame_file_path = os.path.join(output_dir, filename)
        with open(frame_file_path, 'w') as framefile:
            framefile.write(content_to_write)
        print(f"CSV文件 '{filepath}' 和帧内容文件 '{frame_file_path}' 写入成功！")

    def save_full_index_cache(self):
        """
        Writes a CSV file from the given index dictionary.
        Args:
            index (dict): The index dictionary with numerical keys and inner dictionaries as values.
            output_dir (str): The directory to save the CSV file.
            filename (str): The name of the CSV file.
        """
        output_dir = fr"/home/wangshuo_20/pythonpr/VDBMS_ws/media/videoCacheData/{self.config.video_name}/seiden"
        # output_dir = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden"
        filename = fr"{self.config.video_name}_cache_full.csv"
        index = self.full_index_cache
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Construct the full file path
        filepath = os.path.join(output_dir, filename)
        # Open the CSV file for writing
        # print("[write csv index_cache]",index)
        with open(filepath, 'w', newline='') as csvfile:
            # Define the column headers
            fieldnames = ['frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'ind']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write the header row
            writer.writeheader()
            # Iterate through the index dictionary and write rows
            for key, inner_dicts in index.items():
                # Create a dictionary with the required structure
                for inner_dict in inner_dicts:
                    row_dict = {
                        'frame': key,
                        'object_name': inner_dict.get('object_name'),
                        'confidence': inner_dict.get('confidence'),
                        'xmin': inner_dict.get('xmin'),
                        'ymin': inner_dict.get('ymin'),
                        'xmax': inner_dict.get('xmax'),
                        'ymax': inner_dict.get('ymax'),
                        'ind': key  # Use the key as the 'ind' value
                    }
                    # Write the row to the CSV file
                    writer.writerow(row_dict)
        # 下面对缓存帧保存，用于后续对该视频的重复查询--这里先简化到对同一查询语句的复用
        # 获取排序后的帧号列表（整数类型）
        sorted_keys = sorted(index.keys())
        print('[len_cache_frame]: ', len(sorted_keys))
        content_to_write = '\n'.join(str(key) for key in sorted_keys)
        output_dir = fr"/home/wangshuo_20/pythonpr/VDBMS_ws/media/videoCacheData/{self.config.video_name}/seiden"
        # output_dir = fr"/home/wangshuo_20/pythonpr/thesis_data/tasti_data/cache/{self.config.video_name}/seiden"
        # output_dir = fr"D:/Projects/PyhtonProjects/thesis/tasti_data/cache/{self.config.video_name}/seiden"
        filename = fr"{self.config.video_name}_reps_frame.txt"
        frame_file_path = os.path.join(output_dir, filename)
        with open(frame_file_path, 'w') as framefile:
            framefile.write(content_to_write)
        print(f"CSV文件 '{filepath}' 和帧内容文件 '{frame_file_path}' 写入成功！")

    def clear_cache(self):
        del self.index_cache
        self.index_cache = {}

    def inference_transforms(self):
        ttransforms = torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((320, 320)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return ttransforms

    ##### Unimportant functions ######

    def get_cache_dir(self):
        os.makedirs(self.config.cache_dir, exist_ok=True)
        return self.config.cache_dir

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        root = '/home/wangshuo_20/pythonpr/VDBMS_ws/media'
        # root = '/home/wangshuo_20/pythonpr/thesis_data/video_data'
        # root = 'D:/Projects/PyhtonProjects/thesis/video_data'
        ROOT_DATA = self.config.video_name
        category = self.config.category
        print(category)
        # Define the columns for the CSV file
        columns = ['frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'ind']
        if category != 'car':
            labels_fp = os.path.join(root, ROOT_DATA, f'tasti_labels_{category}.csv')
        else:
            labels_fp = os.path.join(root, ROOT_DATA, 'tasti_labels.csv')
        # 检查目录是否存在，如果不存在则创建
        labels_dir = os.path.dirname(labels_fp)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        if not os.path.exists(labels_fp):
            # Create an empty DataFrame with specified columns
            empty_df = pd.DataFrame(columns=columns)
            # Write the DataFrame to CSV
            empty_df.to_csv(labels_fp, index=False)

        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache),
            category=self.config.category
        )
        return labels

    def get_target_dnn_dataset(self, train_or_test):
        ### just convert the loaded data into a dataset.
        dataset = InferenceDataset(self.images)
        return dataset

    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model

    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        # 将模型的全连接层（fc）替换为一个具有输入特征数量为512和输出特征数量为128的线性层，以更改模型的输出维度。
        model.fc = torch.nn.Linear(512, 128)
        return model

    def get_embedding_dnn_dataset(self, train_or_test):
        dataset = InferenceDataset(self.images)
        return dataset

    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        # 将模型的全连接层（fc）替换为一个恒等函数（Identity），即不做任何改变，以保持模型的预训练状态。最后，返回修改后的模型对象
        model.fc = torch.nn.Identity()
        return model

    # wser write -- get_confidence_of_yolvo5s_result
    def get_confidence_of_yolvo5s_result(self, results):
        # print('[results]: ',results)
        max_person_confidence = 0.0
        # 遍历检测结果中的每个预测框
        for row in results:
            # 如果预测框的类别是 person
            if row['object_name'] == 'car':  # person 类别的索引为 0
                # 更新最高置信度的值
                max_person_confidence = max(max_person_confidence, row['confidence'])
        # 输出最高置信度的 person 类别预测框的置信度
        # print("最高置信度的 car 类别预测框的置信度:", max_person_confidence)
        return max_person_confidence
