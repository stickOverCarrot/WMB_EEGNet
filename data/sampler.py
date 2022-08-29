import torch as th
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
import copy
import random


class BalanceIDSampler(Sampler):
    def __init__(self, source_X, target_X, source_y, target_y, batch_size):
        super(BalanceIDSampler, self).__init__(target_X)
        self.source_X = source_X
        self.target_X = target_X
        self.source_y = source_y
        self.target_y = target_y
        self.num_cls = len(set(target_y))
        self.num_source = len(source_X)
        self.length = max(len(target_X), max([len(d) for d in source_X]))
        self.num_batch_per_cls = batch_size // (self.num_source + 1) // self.num_cls
        self.num_batch_per_ids = self.num_batch_per_cls * self.num_cls
        self.batch_size = self.num_batch_per_ids * (self.num_source + 1)
        self.ids_cls, self.statistics, self.data_num = self._count()
        self.length = self._cal_length()

    def _count(self):
        ids_cls = {}
        statistics = []
        for i in range(self.num_source):
            ids_cls[i] = {}
            statistics.append([])
            for j in range(self.num_cls):
                ids_cls[i][j] = np.where(j == self.source_y[i])[0]
                statistics[i].append(len(ids_cls[i][j]))
            statistics[i].append(sum(statistics[i]))
        ids_cls[self.num_source] = {}
        statistics.append([])
        for i in range(self.num_cls):
            ids_cls[self.num_source][i] = np.where(i == self.target_y)[0]
            statistics[self.num_source].append(len(ids_cls[self.num_source][i]))
        statistics[self.num_source].append(sum(statistics[self.num_source]))
        statistics = np.asarray(statistics)
        data_num = np.sum(statistics[:, -1])
        return ids_cls, statistics, data_num

    def _prepare(self):
        batch_idx_dict = defaultdict(list)
        last_batch_list = []
        cumsum = np.cumsum(self.statistics[:, -1])
        cumsum = np.concatenate([np.asarray([0]), cumsum])
        max_mount_ids = 0
        for i in range(self.num_source + 1):
            cls_list = []
            max_ = 0
            for j in range(self.num_cls):
                cls_list.append(copy.deepcopy(self.ids_cls[i][j]) + cumsum[i])
                if max_ < len(cls_list[j]):
                    max_ = len(cls_list[j])
            for j in range(self.num_cls):
                if len(cls_list[j]) < max_:
                    cls_list[j] = np.random.choice(cls_list[j], size=max_, replace=True)
                random.shuffle(cls_list[j])
            k = 0
            while k < max_:
                batch = []
                if k + self.num_batch_per_cls < max_:
                    for j in range(self.num_cls):
                        batch.extend(cls_list[j][k:k + self.num_batch_per_cls])
                else:
                    last_batch = []
                    for j in range(self.num_cls):
                        batch.extend(cls_list[j][k:])
                        last_batch.extend(cls_list[j][k:])
                    length = (self.num_batch_per_ids - len(last_batch)) // self.num_cls
                    if length > 0:
                        for j in range(self.num_cls):
                            idx = np.random.choice(cls_list[j], size=length, replace=False)
                            last_batch.extend(idx)
                    assert self.num_batch_per_ids == len(last_batch)
                    random.shuffle(last_batch)
                    last_batch_list.append(last_batch)
                random.shuffle(batch)
                k += self.num_batch_per_cls
                batch_idx_dict[i].append(batch)
            if max_mount_ids < len(batch_idx_dict[i]):
                max_mount_ids = len(batch_idx_dict[i])
        # last_batch_len = []
        for i in range(self.num_source + 1):
            last_batch = batch_idx_dict[i][-1]
            # last_batch_len.append(len(last_batch))
            if len(batch_idx_dict[i]) < max_mount_ids:
                batch_idx = copy.deepcopy(batch_idx_dict[i][:-1])
                idx = np.random.choice([i for i in range(len(batch_idx_dict[i]) - 1)],
                                       size=max_mount_ids - len(batch_idx_dict[i]),
                                       replace=(len(batch_idx_dict[i]) - 1) < (max_mount_ids - len(batch_idx_dict[i])))
                for j in idx:
                    batch_idx.append(batch_idx_dict[i][j])
                batch_idx.append(last_batch)
                batch_idx_dict[i] = batch_idx
        # if len(set(last_batch_len)) != 1:
        for i in range(self.num_source + 1):
            batch_idx_dict[i][-1] = last_batch_list[i]
        return batch_idx_dict, max_mount_ids

    def _cal_length(self):
        batch_idx_dict, max_mount_ids = self._prepare()
        length = 0
        for i in range(max_mount_ids):
            for j in range(self.num_source + 1):
                length += len(batch_idx_dict[j][i])
        return length

    @property
    def revised_batch_size(self):
        return self.batch_size

    def __iter__(self):
        batch_idx_dict, max_mount_ids = self._prepare()
        outputs = []
        for i in range(max_mount_ids):
            for j in range(self.num_source + 1):
                outputs.extend(batch_idx_dict[j][i])

        return iter(outputs)

    def __len__(self):
        return self.length


class BalancedClassSampler(Sampler):
    def __init__(self, target_X, target_y, batch_size):
        super(BalancedClassSampler, self).__init__(target_X)
        self.target_X = target_X
        self.target_y = target_y
        self.num_cls = len(set(target_y))
        self.length = len(target_X)
        self.num_batch_per_cls = batch_size // self.num_cls
        self.batch_size = self.num_batch_per_cls * self.num_cls
        self.ids_cls, self.statistics = self._count()
        self.length = self._cal_length()

    def _count(self):
        ids_cls = {}
        statistics = []
        for j in range(self.num_cls):
            ids_cls[j] = np.where(j == self.target_y)[0]
            statistics.append(len(ids_cls[j]))
        statistics.append(sum(statistics))
        statistics = np.asarray(statistics)
        return ids_cls, statistics

    def _prepare(self):
        batch_idx = []

        cls_list = []
        max_ = 0
        for j in range(self.num_cls):
            cls_list.append(copy.deepcopy(self.ids_cls[j]))
            if max_ < len(cls_list[j]):
                max_ = len(cls_list[j])
        for j in range(self.num_cls):
            if len(cls_list[j]) < max_:
                cls_list[j] = np.random.choice(cls_list[j], size=max_, replace=True)
            random.shuffle(cls_list[j])
        k = 0
        while k < max_:
            batch = []
            if k + self.num_batch_per_cls < max_:
                for j in range(self.num_cls):
                    batch.extend(cls_list[j][k:k + self.num_batch_per_cls])
            else:
                for j in range(self.num_cls):
                    batch.extend(cls_list[j][k:])
            random.shuffle(batch)
            k += self.num_batch_per_cls
            batch_idx.extend(batch)
        return batch_idx

    def _cal_length(self):
        batch_idx = self._prepare()
        return len(batch_idx)

    @property
    def revised_batch_size(self):
        return self.batch_size

    def __iter__(self):
        batch_idx = self._prepare()
        return iter(batch_idx)

    def __len__(self):
        return self.length
