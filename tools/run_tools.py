import torch as th
import torch.nn.functional as F
import math
import sys
import time
import datetime
import collections
import numpy as np
from .utils import AverageMeter, accuracy, lr_change_over_epoch1, lr_change_over_epoch2


def train_one_epoch(dataLoader, model, device, opt, cls_criterion1, cls_criterion2,
                    start_time, epoch, args):
    print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion1 = cls_criterion1.to(device)
    cls_criterion2 = cls_criterion2.to(device)
    source_id_list = args.source_id_list
    target_id = args.target_id
    num_source = len(source_id_list)
    dict_log = {'loss': AverageMeter(), 'last_cls_loss': AverageMeter(), 'acc': AverageMeter()}
    for i in source_id_list:
        dict_log[i] = {'s_cls_loss': AverageMeter(), 't_cls_loss': AverageMeter(),
                       's_acc': AverageMeter(), 't_acc': AverageMeter()}

    if 1 == args.adjust_lr:
        lr_change_over_epoch1(opt, args.lr, epoch, args.epochs)
    elif 2 == args.adjust_lr:
        lr_change_over_epoch2(opt, args.lr, epoch)
    model.train()
    for step, (features, labels) in enumerate(dataLoader):
        batch_size_s = len(features) // (num_source + 1)
        features = features.to(device)
        labels = labels[:, 0].to(device)
        labels_list = [labels[i * batch_size_s: (i + 1) * batch_size_s] for i in range(num_source + 1)]
        opt.zero_grad()

        s_logits_list, t_logits_list, cls = model(features, is_target_only=False)

        s_cls_loss_list = []
        s_cls_acc_list = []
        t_cls_loss_list = []
        t_cls_acc_list = []
        for i, (s_logits, t_logits, s_labels) in enumerate(zip(s_logits_list, t_logits_list, labels_list[:num_source])):
            s_cls_loss_list.append(cls_criterion1(s_logits, s_labels))
            dict_log[source_id_list[i]]['s_cls_loss'].update(s_cls_loss_list[i].item(), len(s_logits))
            s_cls_acc_list.append(accuracy(s_logits.detach(), s_labels.detach())[0])
            dict_log[source_id_list[i]]['s_acc'].update(s_cls_acc_list[i].item(), len(s_logits))
            t_cls_loss_list.append(cls_criterion1(t_logits, labels_list[num_source]))
            dict_log[source_id_list[i]]['t_cls_loss'].update(t_cls_loss_list[i].item(), len(labels_list[num_source]))
            t_cls_acc_list.append(accuracy(t_logits.detach(), labels_list[num_source].detach())[0])
            dict_log[source_id_list[i]]['t_acc'].update(t_cls_acc_list[i].item(), len(labels_list[num_source]))

        classifier_loss = cls_criterion2(th.log(cls), labels_list[num_source].detach())
        loss = 0.5 * (th.stack(s_cls_loss_list).mean() + th.stack(t_cls_loss_list).mean())

        loss = loss + classifier_loss
        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        dict_log['last_cls_loss'].update(classifier_loss.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))
        loss.backward()
        opt.step()
        acc = accuracy(cls.detach(), labels_list[num_source].detach())[0]
        dict_log['acc'].update(acc.item(), len(cls))
        if 0 == (step + 1) % args.print_freq:
            lr = list(opt.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\tlr:{}\t'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader), lr)
            for key in source_id_list:
                value = dict_log[key]
                key = str(key)

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_cls_loss/' + key,
                                                                 value['s_cls_loss'].val, value['s_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('s_acc/' + key,
                                                                value['s_acc'].val, value['s_acc'].avg)
                print_information += acc_info

                loss_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_cls_loss/' + key + '_{}'.format(target_id),
                                                                 value['t_cls_loss'].val, value['t_cls_loss'].avg)
                print_information += loss_info

                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/' + key + '_{}'.format(target_id),
                                                                value['t_acc'].val, value['t_acc'].avg)
                print_information += (acc_info + '\n')

            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('classifier_loss',
                                                             dict_log['last_cls_loss'].val,
                                                             dict_log['last_cls_loss'].avg)
            print_information += loss_info

            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('all_loss',
                                                             dict_log['loss'].val, dict_log['loss'].avg)
            print_information += loss_info
            loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('acc',
                                                             dict_log['acc'].val, dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End training At Epoch:{}--------------------------'.format(epoch + 1))


def evaluate_one_epoch(dataLoader, model, device, cls_criterion, start_time, epoch, args):
    print('--------------------------Start Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    cls_criterion = cls_criterion.to(device)

    target_id = args.target_id
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter(), 'together_acc': AverageMeter()}
    for i in args.source_id_list:
        dict_log[i] = {'t_acc': AverageMeter()}
    model.eval()
    for step, (features, labels) in enumerate(dataLoader):
        features = features.to(device)
        labels = labels[:, 0].to(device)
        with th.no_grad():
            _, all_preds_list, preds = model(features)
            loss = cls_criterion(th.log(preds), labels)
            if len(loss.shape) > 0:
                loss = loss.mean()

        dict_log['loss'].update(loss.item(), len(features))
        acc = accuracy(preds.detach(), labels.detach())[0]
        # acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        for i in range(len(args.source_id_list)):
            acc = accuracy(all_preds_list[i].detach(), labels.detach())[0]
            dict_log[args.source_id_list[i]]['t_acc'].update(acc.item(), len(features))
        if len(all_preds_list) > len(args.source_id_list):
            acc = accuracy(all_preds_list[-1].detach(), labels.detach())[0]
            dict_log['together_acc'].update(acc.item(), len(features))
        if (step + 1) == len(dataLoader):
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]

            print_information = 'epoch:{}/{}\ttime consumption:{}\tstep:{}/{}\t\n'.format(
                epoch + 1, args.epochs, et, step, len(dataLoader))
            for key in args.source_id_list:
                value = dict_log[key]
                key = str(key)
                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/' + key + '_{}'.format(target_id),
                                                                value['t_acc'].val, value['t_acc'].avg)
                print_information += (acc_info + '\n')
            if len(all_preds_list) > len(args.source_id_list):
                acc_info = "{}(val/avg):{:.3f}/{:.3f}\t".format('t_acc/{}'.format(target_id),
                                                                dict_log['together_acc'].val,
                                                                dict_log['together_acc'].avg)
                print_information += acc_info + '\n'
            loss_info = "id:{}\t{}(val/avg):{:.3f}/{:.3f}\t{}(val/avg):{:.3f}/{:.3f}\n ".format(target_id,
                                                                                                'loss',
                                                                                                dict_log['loss'].val,
                                                                                                dict_log['loss'].avg,
                                                                                                'acc',
                                                                                                dict_log['acc'].val,
                                                                                                dict_log['acc'].avg)
            print_information += loss_info
            print(print_information)

    print('--------------------------End Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg
