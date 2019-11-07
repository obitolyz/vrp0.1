# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 20:12
# @Author  : obitolyz
# @FileName: Data_Generator.py
# @Software: PyCharm

import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class VRPDataset(Dataset):
    def __init__(self, node_num, num_samples=1000000, random_seed=111):
        # service_num: node_num - 1
        super(VRPDataset, self).__init__()
        random.seed(random_seed)
        self.dataset = []

        for _ in tqdm(range(num_samples), disable=False):
            sample = list()
            sample.append(torch.FloatTensor([random.uniform(0, 1), random.uniform(0, 1), 0, 0, 0]))
            for i in range(node_num - 1):
                x, y = random.uniform(0, 1), random.uniform(0, 1)
                capacity = random.randint(1, 10)
                t1 = random.uniform(0, 5)
                t2 = t1 + 2
                sample.append(torch.FloatTensor([x, y, capacity, t1, t2]))

            self.dataset.append(torch.cat(sample, 0).view(-1, 5))

        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]
