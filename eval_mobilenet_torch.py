'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import argparse

from torch.autograd import Variable

from models import mobilenet_v1, mobilenet_v2
from utils import measure_model, AverageMeter, progress_bar, accuracy, process_state_dict

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='mobilenet_0.5flops', type=str, help='name of the model to test')
parser.add_argument('--imagenet_path', default=None, type=str, help='Directory of ImageNet')
parser.add_argument('--n_gpu', default=1, type=int, help='name of the job')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()


def get_dataset():
    # lazy import
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    if not args.imagenet_path:
        raise Exception('Please provide valid ImageNet path!')
    print('=> Preparing data..')
    valdir = os.path.join(args.imagenet_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    input_size = 224
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_worker, pin_memory=True)
    n_class = 1000
    return val_loader, n_class


def get_model(n_class):
    print('=> Building model {}...'.format(args.model))
    if args.model == 'mobilenet_0.5flops':
        net = mobilenet_v1.MobileNet(n_class, profile='0.5flops')
        checkpoint_path = './checkpoints/torch/mobilenet_imagenet_0.5flops_70.5.pth.tar'
    elif args.model == 'mobilenet_0.5time':
        net = mobilenet_v1.MobileNet(n_class, profile='0.5time')
        checkpoint_path = './checkpoints/torch/mobilenet_imagenet_0.5time_70.2.pth.tar'
    elif args.model == 'mobilenetv2':
        net = mobilenet_v2.MobileNetV2(n_class, profile='normal')
        checkpoint_path = './checkpoints/torch/mobilenetv2_imagenet_71.814.pth.tar'
    elif args.model == 'mobilenetv2_0.7flops':
        net = mobilenet_v2.MobileNetV2(n_class, profile='0.7flops')
        checkpoint_path = './checkpoints/torch/mobilenetv2_imagenet_0.7amc_70.854.pth.tar'
    else:
        raise NotImplementedError

    print('=> Loading checkpoints..')
    checkpoint = torch.load(checkpoint_path)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']  # get state_dict
    net.load_state_dict(process_state_dict(checkpoint))  # remove .module

    return net


def evaluate():
    # build dataset
    val_loader, n_class = get_dataset()
    # build model
    net = get_model(n_class)  # for measure
    n_flops, n_params = measure_model(net, 224, 224)
    print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M'.format(n_params / 1e6, n_flops / 1e6))
    del net
    net = get_model(n_class)

    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, list(range(args.n_gpu)))
        cudnn.benchmark = True

    # begin eval
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(val_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))


if __name__ == '__main__':
    evaluate()
