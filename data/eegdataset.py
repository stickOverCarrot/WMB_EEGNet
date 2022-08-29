import torch as th
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
import numpy as np
import random


class EEGDataset(Dataset):
    def __init__(self, source_X_list, target_X, source_y_list, target_y, data_num=None, data_augment=None):
        super(EEGDataset, self).__init__()
        self.data_num = data_num
        if source_X_list is not None:
            self.num_cls = len(set(source_y_list[0]))
            self.length = sum([len(source_X_list[i])for i in range(len(source_X_list))])
            self.num_source = len(source_X_list)
            if target_X is not None:
                self.X = np.concatenate([np.concatenate(source_X_list), target_X])
                self.y = np.concatenate([np.concatenate(source_y_list), target_y])
            else:
                self.X = np.concatenate(source_X_list)
                self.y = np.concatenate(source_y_list)
        elif target_X is not None:
            self.num_cls = len(set(target_y))
            self.length = len(target_X)
            self.X = target_X
            self.y = target_y
        else:
            raise ValueError("It is at least one of source_X_list and target_X is not None")

    def __len__(self):
        if self.data_num is not None:
            return self.data_num
        return self.length

    def __getitem__(self, item):
        return th.FloatTensor(self.X[item]), th.LongTensor([self.y[item]])


# def gauss_data_augmentation(data, sigma=0.01, p=0.5):
#     if random.random() > p:
#         return data
#     gauss_data = np.zeros_like(data)
#     #             print(data.shape)
#     for i in range(data.shape[0]):
#         ch = data[i]
#         pch = ch + np.random.normal(loc=0.0, scale=sigma, size=ch.size())
#         gauss_data[i] = pch
#     return gauss_data


