import torch as th
import os
import sys
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader

from tools.utils import set_seed, set_save_path, Logger, save, EarlyStopping
from tools.run_tools import train_one_epoch, evaluate_one_epoch
from models.EEGNet import WMB_EEGNet
from data.bciciv2a_process import load_bciciv2a_data_single_subject
from data.high_gamma_process import load_highgamma_data_single_subject
from data.bciciv2b_process import load_bciciv2b_data_single_subject
from data.sampler import BalanceIDSampler
from data.eegdataset import EEGDataset


def train(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, 'information.txt'))
    start_epoch = 0
    # ------------------------------------------------device setting--------------------------------------------------
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'

    # ------------------------------------------------data setting----------------------------------------------------
    if "high_gamma" in args.data_path:
        args.sub_num = 14
        args.class_num = 4
        load_data = load_highgamma_data_single_subject
    elif "BCICIV_2a" in args.data_path:
        args.sub_num = 9
        args.class_num = 4
        load_data = load_bciciv2a_data_single_subject
    elif "BCICIV_2b" in args.data_path:
        args.sub_num = 9
        args.class_num = 2
        load_data = load_bciciv2b_data_single_subject
    else:
        raise ValueError("only support high_gamma or BCICIV_2a dataset.")
    id_list = [i + 1 for i in range(args.sub_num)]
    source_id_list = []
    source_X_list, source_y_list = [], []
    target_X, target_y, target_test_X, target_test_y = [None] * 4
    for i in id_list:
        if i != args.target_id:
            train_X, train_y, _, _ = load_data(args.data_path, subject_id=i, to_tensor=False)
            source_id_list.append(i)
            source_X_list.append(train_X)
            source_y_list.append(train_y)
        else:
            target_X, target_y, target_test_X, target_test_y = load_data(args.data_path, subject_id=i, to_tensor=False)
    args.source_id_list = source_id_list

    data_sampler = BalanceIDSampler(source_X_list, target_X, source_y_list, target_y, args.batch_size)
    args.batch_size = data_sampler.revised_batch_size
    data_num = data_sampler.__len__()
    train_data = EEGDataset(source_X_list, target_X, source_y_list, target_y, data_num)
    trainLoader = DataLoader(train_data, args.batch_size, shuffle=False, sampler=data_sampler, num_workers=8,
                             drop_last=False)
    test_data = EEGDataset(source_X_list=None, target_X=target_test_X, source_y_list=None, target_y=target_test_y)
    testLoader = DataLoader(test_data, args.batch_size // (len(source_id_list) + 1),
                            shuffle=False, num_workers=4, drop_last=False)
    # ------------------------------------------------model setting----------------------------------------------------

    backbone = WMB_EEGNet(source_X_list[0].shape[-2], source_X_list[0].shape[-1], args.class_num,
                          pool_mode=args.pool, f1=args.f1, d=2, f2=args.f1 * 2, kernel_length=64,
                          drop_prob=args.dropout, source_num=len(source_id_list))

    print("---------------------------------configuration information-------------------------------------------------")
    for i in list(vars(args).keys()):
        print("{}:{}".format(i, vars(args)[i]))
    # -----------------------------------------------training setting--------------------------------------------------
    opt = th.optim.Adam(backbone.parameters(), lr=args.lr, weight_decay=args.w_decay)
    cls_criterion1 = th.nn.CrossEntropyLoss()
    cls_criterion2 = th.nn.NLLLoss()
    if args.early_stop:
        if "high_gamma" in args.data_path:
            stop_train = EarlyStopping(patience=80, max_epochs=args.epochs)
        else:
            stop_train = EarlyStopping(patience=160, max_epochs=args.epochs)
    # -----------------------------------------------resume setting--------------------------------------------------
    best_acc = 0
    # -------------------------------------------------run------------------------------------------------------------
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.early_stop and stop_train.early_stop:
            print("early stop in {}!".format(epoch))
            break
        steps = train_one_epoch(trainLoader, backbone, device, opt, cls_criterion1,
                                cls_criterion2,
                                start_time, epoch, args)

        avg_acc, avg_loss = evaluate_one_epoch(testLoader, backbone, device, cls_criterion2, start_time,
                                               epoch, args)

        if args.early_stop:
            stop_train(avg_acc)
        save_checkpoints = {'model_classifier': backbone.state_dict(),
                            'epoch': epoch,
                            'steps': steps,
                            'acc': avg_acc}
        if avg_acc > best_acc:
            best_acc = avg_acc
            save(save_checkpoints, os.path.join(args.model_classifier_path, 'model_best.pth.tar'))
        print('best_acc:{}'.format(best_acc))
        save(save_checkpoints, os.path.join(args.model_classifier_path, 'model_newest.pth.tar'))
        if 1.0 == best_acc:
            print("The modal has achieved 100% acc! Early stop at epoch:{}!".format(epoch))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default='/home/wong/dataset_ubuntu/BCICIV_2a_gdf/no_preprocess_05_100_4500',
                        help='Data path.')
    parser.add_argument('-f1', type=int, default=16,
                        help='the number of filters in EEGNet.')
    parser.add_argument('-target_id', type=int, default=2, help='Target id.')
    parser.add_argument('-dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('-pool', type=str, default='mean', choices=['max', 'mean'])
    parser.add_argument('-epochs', type=int, default=800, help='Number of epochs to train.')
    parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-adjust_lr', type=int, default=1, choices=[0, 1, 2], help='Learning rate changes over epoch.')
    parser.add_argument('-w_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-batch_size', default=576, type=int,
                        help='batch size for training')
    parser.add_argument('-early_stop', action='store_false', help='Train early stop.')
    parser.add_argument('-print_freq', type=int, default=3, help='The frequency to show training information.')
    parser.add_argument('-father_path', type=str, default='save',
                        help='The father path of models parameters, log files.')
    parser.add_argument('-seed', type=int, default='111', help='Random seed.')
    args_ = parser.parse_args()
    train(args_)
