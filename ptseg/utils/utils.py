'''
Misc Utility functions
'''

import os
import torch
from collections import OrderedDict
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("unsupported value")


def save_checkpoint(state, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, filename)
    torch.save(state, path + '.pth')


def get_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_formal_accuracy(output, target):
    batch_size = target.size(0)
    _, pred = torch.max(output, 1)
    correct = 0
    correct += torch.sum(pred.data == target.data)
    acc = torch.FloatTensor(1)
    acc[0] = float(correct) / batch_size * 100.0
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    #    return optimizer
    # if (iter) % 50000 == 0:
    # lr = init_lr * 0.1 ** ((iter) // 50000)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """   
    new_state_dict = OrderedDict() 
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        # del new_state_dict[k]
    return new_state_dict


def convert_state_dict_for_seresnet(state_dict):
    new_state_dict = OrderedDict()
    for name, v in state_dict.items():
        if 'layer0' in name:
            new_state_dict[name[7:]] = v
        elif 'last_linear.weight' in name:
            new_state_dict['fc.weight'] = v
        elif 'last_linear.bias' in name:
            new_state_dict['fc.bias'] = v
        else:
            new_state_dict[name] = v
    for name, v in new_state_dict.items():
        if 'fc' in name and 'weight' in name:
            # print(name, ' size changed to x,y,1,1')
            new_v = v
            new_v = new_v.view(new_v.shape[0], new_v.shape[1])
            # print(new_v.shape)
            new_state_dict[name] = new_v
    return new_state_dict

