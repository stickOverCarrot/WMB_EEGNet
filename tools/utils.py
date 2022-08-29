import numpy as np
import torch as th
import random
import errno
import os
import sys
import time
import math
from sklearn.utils import check_random_state


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args):
    father_path = os.path.join(father_path, '{}'.format(time.strftime("%m_%d_%H_%M")))
    # try:
    #     father_path = os.path.join(father_path, args.model + '_{}'.format(args.f1) + '_{}'.format(args.dg),
    #                                str(args.target_id))
    # except:
    #     father_path = os.path.join(father_path, args.model + '_{}'.format(args.f1), str(args.target_id))
    mkdir(father_path)
    args.log_path = father_path
    args.tensorboard_path = os.path.join(father_path, 'tensorboard')
    args.model_adj_path = father_path
    args.model_classifier_path = father_path
    return args


def save(checkpoints, save_path):
    th.save(checkpoints, save_path)


def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    if shape:
        target = target.view(shape)
    return ret


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_change_over_epoch1(opt, init_value, epoch, all_epoch, min_value=None, max_value=None):
    if min_value is not None and max_value is None:
        assert min_value <= max_value
    p = epoch / all_epoch
    value = init_value / math.pow(1 + 10 * p, 0.75)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(min_value, value)
    for param in opt.param_groups:
        param['lr'] = value


def lr_change_over_epoch2(opt, init_value, epoch, epochs):
    if epoch < epochs / 5:
        lr = init_value * 5 * epoch / epochs
    else:
        lr = init_value * 0.5 * (1 + math.cos(180 * (epoch - epochs / 5) / (epochs / 4)))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


class EarlyStopping(object):
    """
    Early stops the training if validation loss
    doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, max_epochs=80):
        """
        patience (int): How long to wait after last time validation
        loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation
        loss improvement.
                        Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # my addition:
        self.max_epochs = max_epochs
        self.max_epoch_stop = False
        self.epoch_counter = 0
        self.should_stop = False
        self.checkpoint = None

    def __call__(self, val_loss):
        # my addition:
        self.epoch_counter += 1
        if self.epoch_counter >= self.max_epochs:
            self.max_epoch_stop = True

        score = val_loss

        if self.best_score is None:
            print('')
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} '
                  f'out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        # my addition:
        if any([self.max_epoch_stop, self.early_stop]):
            self.should_stop = True


def gaussian_noise(X, y, std, random_state=None, p=0.5):
    if p > random.random():
        return X, y
    rng = check_random_state(random_state)
    if isinstance(std, th.Tensor):
        std = std.to(X.device)
    noise = th.from_numpy(
        rng.normal(
            loc=np.zeros(X.shape),
            scale=1
        ),
    ).float().to(X.device) * std
    transformed_X = X + noise
    return transformed_X, y


def mixup(X, y, alpha, p=0.5):
    device = X.device
    batch_size, n_channels, n_times = X.shape
    idx_perm = th.randperm(batch_size).to(device)

    X_mix = th.zeros((batch_size, n_channels, n_times)).to(device)
    y_a = th.arange(batch_size).to(device)
    y_b = th.arange(batch_size).to(device)

    if alpha > 0:
        lam = th.from_numpy(np.random.beta(alpha, alpha, size=[batch_size, ])).to(device)
        # lam = th.tensor(np.random.beta(alpha, alpha)).to(device)
    else:
        lam = th.ones(size=[batch_size, ], dtype=th.float32).to(device)
        # lam = th.tensor(1.).to(device)

    for idx in range(batch_size):
        X_mix[idx] = lam[idx] * X[idx] + (1 - lam[idx]) * X[idx_perm[idx]]
        # X_mix[idx] = lam * X[idx] + (1 - lam) * X[idx_perm[idx]]
        y_a[idx] = y[idx]
        y_b[idx] = y[idx_perm[idx]]

    return X_mix, (y_a, y_b, lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    if len(loss.shape) > 0:
        loss = loss.mean()
    return loss
