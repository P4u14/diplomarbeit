# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:53:30 2017
@author: bbrattol
Modified by: Paula Schreiber
"""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable


def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def configure_device(args):
    if args.gpu >= 0 and torch.backends.mps.is_available():
        device = torch.device("mps")
        log_print("MPS available. Training runs on MPS.", log_dir=args.checkpoint)
    elif args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        log_print(('Using GPU %d' % args.gpu), log_dir=args.checkpoint)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        device = torch.device("cpu")
        log_print('Training läuft auf CPU.', log_dir=args.checkpoint)
    return device

def load_data(args, jigsaw_dataloader, train=True):
    if train:
        suffix = 'train'
    else:
        suffix = 'val'
    img_path = args.data + '/ILSVRC2012_img_' + suffix

    if os.path.exists(img_path + '_255x255'):
        img_path += '_255x255'
    img_data = jigsaw_dataloader(img_path, args.data + f'/ilsvrc12_{suffix}.txt', classes=args.classes)
    img_loader = torch.utils.data.DataLoader(dataset=img_data, batch_size=args.batch, shuffle=True, num_workers=args.cores)
    return img_data, img_loader

def load_network(args, net):
    # load from checkpoint if exists, otherwise from model
    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files) > 0:
            files.sort()
            # print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint + '/' + ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            log_print('Starting from: ', ckp, log_dir=args.checkpoint)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)

def log_print(*text, log_dir, sep=' ', end='\n', file=None, flush=False):
    # standard output
    print(*text, sep=sep, end=end, file=file, flush=flush)
    # write in log file in checkpoint folder
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'logs.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        print(*text, sep=sep, end=end, file=f)

def plot_train_and_val_metrics(args, train_accs, train_losses, val_accs, val_losses):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.checkpoint, 'training_metrics_plot.png')
    plt.savefig(plot_path)
    log_print(f"Training metrics plot saved to {plot_path}", log_dir=args.checkpoint)
    plt.close()

def save_final_model(args, net):
    final_model_path = os.path.join(args.checkpoint, 'final_model.pth.tar')
    net.save(final_model_path)
    log_print(f"Final model saved to {final_model_path}", log_dir=args.checkpoint)
