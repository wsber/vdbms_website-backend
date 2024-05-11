import cv2
import swag
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm
import pandas as pd

class TripletDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,
            target_dnn_cache,
            list_of_idxs,
            is_close_fn,
            index,
            length=1000,
    ):
        self.dataset = dataset
        self.target_dnn_cache = target_dnn_cache
        self.list_of_idxs = list_of_idxs
        self.is_close_fn = is_close_fn
        self.length = length
        self.index = index

        self.buckets = []
        for idx in tqdm(self.list_of_idxs, desc="Triplet Dataset Init"):
            # label = self.target_dnn_cache[idx]
            label = self.trans_dic_to_panda(self.index.run_oracle_model(idx))
            # print('[label]:',label)
            found = False
            for bucket in self.buckets:
                rep_idx = bucket[0]
                # rep = self.target_dnn_cache[rep_idx]
                rep = self.trans_dic_to_panda(self.index.run_oracle_model(rep_idx))
                if self.is_close_fn(label, rep):
                    bucket.append(idx)
                    found = True
                    break
            if not found:
                self.buckets.append([idx])
        i = 0
        for item in self.buckets:
            print(fr'[buket {i}]',item)
            i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rand = np.random.RandomState(seed=idx)
        rand.randint(0, 100, size=10)
        
        def get_triplet_helper():
            anchor_bucket_idx = rand.randint(0, len(self.buckets))
            anchor_bucket = self.buckets[anchor_bucket_idx]
            negative_bucket_idx = rand.choice(
                    [idx for idx in range(len(self.buckets)) if idx != anchor_bucket_idx]
            )
            negative_bucket = self.buckets[negative_bucket_idx]

            anchor_idx = rand.choice(anchor_bucket)
            positive_idx = rand.choice(anchor_bucket)
            negative_idx = rand.choice(negative_bucket)

            return anchor_idx, positive_idx, negative_idx

        anchor_idx, positive_idx, negative_idx = get_triplet_helper()
        for i in range(200):
            if abs(anchor_idx - positive_idx) > 30:
                break
            else:
                anchor_idx, positive_idx, negative_idx = get_triplet_helper()
        
        anchor = self.dataset[anchor_idx]
        positive = self.dataset[positive_idx]
        negative = self.dataset[negative_idx]
        
        return anchor, positive, negative

    def trans_dic_to_panda(self,label):
        series_list =[]
        for it in label:
            series = pd.Series(it)
            series_list.append(series)
        return series_list