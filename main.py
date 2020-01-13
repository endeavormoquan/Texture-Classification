# -*- coding: utf-8 -*-
"""
created on Mon Dec 30, 2019
@author: Ding Qimin
"""

import argparse
import torch
import time
from datetime import datetime
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter

from torchvision.models.vgg import vgg19_bn, vgg16_bn
from torchvision.models.inception import inception_v3
from ptseg.models.resnet_ori import resnet50, resnet34, resnet101, resnet152
from ptseg.models.SENet.senet.se_resnet import se_resnet152
from ptseg.models.SENet_ori.se_resnet import se_resnet50

from ptseg.utils import *
from ptseg.utils.lr_helper import IterExponentialLR
from ptseg.utils.dataloader import TextureDataset, get_sampler
from ptseg.utils.freeze import freeze_params
from ptseg.utils.LabelSmoothing import CrossEntropyLabelSmooth, LabelSmoothingLoss

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=25, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
parser.add_argument("--arch", default='senet152', type=str, help="model arch")
parser.add_argument('--lr', nargs='?', type=float, default=0.05,
                    help='Learning Rate')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4), 8 machines, each machine 4 workers, total 32')
# parser.add_argument('--dynamic_factor', type=int, default=2)
# parser.add_argument('--group_size', type=int, default=8)
parser.add_argument('--save-dir-path', default='checkpoints/', type=str)
parser.add_argument('--save_dir', nargs='?', type=str, default='./model',
                    help='Path to save model')
parser.add_argument('--resume', nargs='?', type=str, default=None,
                    help='Path to previous saved model to restart from')
parser.add_argument('--start_epoch', type=int, default=-1)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--decay-epoch', default="10, 16, 22", type=lambda x: [int(_) for _ in x.split(',')])
parser.add_argument('--gamma', type=float, default='0.1')
parser.add_argument('--scale-factor', type=float, default='1')
parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--freeze', action='store_false', default=True)
parser.add_argument('--num-classes', type=int, default=47)
parser.add_argument('--resize', type=int, default=384)
parser.add_argument('--crop', type=int, default=320)
parser.add_argument('--note', type=str, default='')

model_zoo = {'resnet50': resnet50,
             'resnet34': resnet34,
             'resnet101': resnet101,
             'resnet152': resnet152,
             'vgg16bn': vgg16_bn,
             'vgg19bn': vgg19_bn,
             'senet152': se_resnet152,
             'se_resnet50': se_resnet50,
             'inception': inception_v3}


def main():
    global args
    args = parser.parse_args()

    if torch.cuda.is_available():
        cudnn.benchmark = True
    # model definition
    if args.arch == 'senet152':
        model = model_zoo[args.arch]()
        state_dict = torch.load('D:\\PycharmWorkspace\\Torch-Texture-Classification\\se_resnet152-d17c99b7.pth')
        new_state_dict = convert_state_dict_for_seresnet(state_dict)
        model.load_state_dict(new_state_dict)
    elif args.arch == 'se_resnet50':
        model = model_zoo[args.arch]()
        state_dict = torch.load('D:\\PycharmWorkspace\\Torch-Texture-Classification\\seresnet50-60a8950a85b2b.pkl')
        model.load_state_dict(state_dict)
    else:
        model = model_zoo[args.arch](pretrained=True)
    if 'resnet' in args.arch:
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif 'vgg' in args.arch:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes)
    elif 'senet' in args.arch:
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif 'se_' in args.arch:
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif 'inception' in args.arch:
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    print(model)
    # exit()
    # resume checkpoint
    checkpoint = None
    if args.resume:
        device = torch.cuda.current_device()
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(device))
        state = convert_state_dict(checkpoint['model_state'])
        model.load_state_dict(state)

    model.cuda()
    model = freeze_params(args.arch, model)
    criterion = nn.CrossEntropyLoss().cuda()

    # no bias decay
    param_optimizer = list(filter(lambda p: p.requires_grad, model.parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.001},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.SGD(optimizer_grouped_parameters,
                                lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # original optimizer
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=args.lr, momentum=0.9, weight_decay=5e-4)
    print('trainable parameters:')
    for param in model.named_parameters():
        if param[1].requires_grad:
            print(param[0])  # [0] name, [1] params
    # params should be a dict or a iterable tensor
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=args.lr, weight_decay=0.01)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # calculated by transforms_utils.py
    normalize = transforms.Normalize(mean=[0.5335619, 0.47571668, 0.4280075], std=[0.26906276, 0.2592897, 0.26745376])
    transform_train = transforms.Compose([
        transforms.Resize(args.resize),  # 384, 256
        transforms.RandomCrop(args.crop),  # 320, 224
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        # transforms.RandomRotation(45),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(args.resize),  # 384
        transforms.RandomCrop(args.crop),  # 320
        transforms.ToTensor(),
        normalize,
    ])

    train_data_root = 'D:\\PycharmWorkspace\\Torch-Texture-Classification\\dataset\\train'
    test_data_root = 'D:\\PycharmWorkspace\\Torch-Texture-Classification\\dataset\\test'
    train_dataset = TextureDataset(train_data_root, train=True, transform=transform_train)
    test_dataset = TextureDataset(test_data_root, train=False, transform=transform_test)
    sampler = get_sampler(train_dataset, args.num_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH, shuffle=False, sampler=sampler,
                              num_workers=0, pin_memory=True)  # num_workers != 0 will broke
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH, shuffle=False,
                             num_workers=0, pin_memory=False)
    print('train data length:', len(train_loader), '. test data length:', len(test_loader), '\n')

    lr_scheduler = MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.gamma, last_epoch=args.start_epoch)

    time_ = time.localtime(time.time())
    log_save_path = 'Logs/' + str(args.arch) + '_' + str(args.lr) + '_' + str(args.BATCH) + \
                    '_' + str(time_.tm_mon) + str(time_.tm_mday) + str(time_.tm_hour) + str(time_.tm_min) + \
                    '.log.txt'
    log_saver = open(log_save_path, mode='w')
    v_args = vars(args)
    for k, v in v_args.items():
        log_saver.writelines(str(k)+' '+str(v) + '\n')
    log_saver.close()

    global writer
    current_time = datetime.now().strftime('%b%d_%H-%M')
    logdir = os.path.join('TensorBoardXLog', current_time)
    writer = SummaryWriter(log_dir=logdir)

    dummy_input = torch.randn(args.BATCH, 3, crop, crop).cuda()
    writer.add_graph(model, dummy_input)
    writer.close()
    exit()
    best_score = 0
    for epoch in range(args.start_epoch + 1, args.EPOCHS):  # args.start_epoch = -1 for MultistepLr
        log_saver = open(log_save_path, mode='a')
        lr_scheduler.step()
        train(train_loader, model, criterion, lr_scheduler, epoch, warm_up=False)
        prec1, loss = validate(test_loader, model, criterion)
        writer.add_scalar('scalar/test_prec', prec1, epoch)
        writer.add_scalar('scalar/test_loss', loss, epoch)
        print('test average is: ', prec1)
        log_saver.writelines('learning rate:' + str(lr_scheduler.get_lr()[0]) +
                             ', epoch:' + str(epoch) +
                             ', test average is: ' + str(prec1) +
                             ', loss average is: ' + str(loss) +
                             '\n')

        save_name = str(args.lr) + '_' + str(args.BATCH)
        save_dir = os.path.join(args.save_dir_path,
                                str(args.arch) + '_' + str(time_.tm_mon) + str(time_.tm_mday) +
                                str(time_.tm_hour) + str(time_.tm_min))
        if prec1 > best_score:
            best_score = prec1
            save_checkpoint({
                'epoch': epoch+1,
                'model_state': model.cpu().state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, save_dir, save_name + '_ckpt_e{}'.format(epoch))
        log_saver.close()
    writer.close()


def train(train_loader, model, criterion, lr_scheduler, epoch, warm_up=False):
    batch_time = AverageMeter()
    loss_accu_ave = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # Loss = CrossEntropyLabelSmooth(args.num_classes, epsilon=0.1)

    model = model.train().cuda()
    for index, (inputs, labels) in enumerate(train_loader):
        if warm_up:
            lr_scheduler.step()
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = torch.squeeze(labels)
        if 'inception' in args.arch:
            output, aux = model(inputs)
        else:
            output = model(inputs)
        loss_accu = criterion(output, labels)
        # loss_accu = Loss(output, labels)
        if args.scale_factor > 1:
            loss_accu = loss_accu * args.scale_factor
        loss_accu_ave.update(loss_accu, inputs.size(0))
        prec1, prec5 = get_accuracy(output, labels, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        model.zero_grad()
        loss_accu.backward()
        lr_scheduler.optimizer.step()
        lr = lr_scheduler.get_lr()[0]
        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] LR:{4:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_Accu {loss_accu.val:.4f} ({loss_accu.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, args.EPOCHS, index, len(train_loader), lr,
                    batch_time=batch_time, loss_accu=loss_accu_ave,
                    top1=top1, top5=top5))
        if index == 0:
            batch_time.reset()
        writer.add_scalar('scalar/train_prec', top1.avg, index+1+epoch*len(train_loader))
        writer.add_scalar('scalar/train_loss', loss_accu_ave.avg, index+1+epoch*len(train_loader))
    writer.add_scalar('scalar/train_prec_epoch', top1.avg, epoch)
    writer.add_scalar('scalar/train_loss_epoch', loss_accu_ave.avg, epoch)


def validate(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Loss = CrossEntropyLabelSmooth(args.num_classes, epsilon=0.1)

    with torch.no_grad():
        model.eval().cuda()
        end = time.time()
        for index, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels = torch.squeeze(labels)
            output = model(inputs)
            loss = criterion(output, labels)
            # loss = Loss(output, labels)
            if args.scale_factor > 1:
                loss = loss * args.scale_factor
            prec1, prec5 = get_accuracy(output.data, labels.cuda(), topk=(1, 5))

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone()
            reduced_prec5 = prec5.clone()

            losses.update(reduced_loss.item(), inputs.size(0))
            top1.update(reduced_prec1.item(), inputs.size(0))
            top5.update(reduced_prec5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       index, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
