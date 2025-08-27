# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:53:30 2017
@author: bbrattol
Modified by: Paula Schreiber
"""
import os

import numpy as np
import torch
import PIL.Image as PILImage
from matplotlib import pyplot as plt

from preprocessing.square_image_preprocessor import SquareImagePreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor


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

def _materialize_preprocessed_cache(base_dir, list_file, cache_dir, preprocessing_steps):
    # Create directory structure
    for line in open(list_file, 'r'):
        rel, *_ = line.strip().split(' ')
        dst_path = os.path.join(cache_dir, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Image preprocessing and saving (only if not already present)
    for line in open(list_file, 'r'):
        rel, *_ = line.strip().split(' ')
        src_path = os.path.join(base_dir, rel)
        dst_path = os.path.join(cache_dir, rel)
        if os.path.exists(dst_path):
            continue

        img = np.asarray(PILImage.open(src_path).convert('RGB'))
        params_stack = []  # Only for logging/debugging
        x = img
        for step in (preprocessing_steps or []):
            x, params = step.preprocess_image(x)
            params_stack.append(params)

        # Transform to uint8 if needed
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)

        PILImage.fromarray(x).save(dst_path)

def load_data(args, jigsaw_dataloader, train=True, preprocessing_steps=None, cache_suffix='_pp'):
    suffix = 'train' if train else 'val'
    base_dir = os.path.join(args.data, f'ILSVRC2012_img_{suffix}')
    list_file = os.path.join(args.data, f'ilsvrc12_{suffix}.txt')
    pp_steps_suffix = ''

    for step in (args.pp or []):
        pp_steps_suffix += f'_{step}'

    if preprocessing_steps:
        # e.g. …/ILSVRC2012_img_train_pp
        log_print(f'Loading preprocessing steps for {suffix}', log_dir=args.checkpoint)
        cache_dir = base_dir + cache_suffix + pp_steps_suffix
        _materialize_preprocessed_cache(base_dir, list_file, cache_dir, preprocessing_steps)
        img_path = cache_dir
    else:
        log_print(f'Loading data without preprocessing steps for {suffix}', log_dir=args.checkpoint)
        img_path = base_dir + '_255x255' if os.path.exists(base_dir + '_255x255') else base_dir

    img_data = jigsaw_dataloader(img_path, list_file, classes=args.classes)
    img_loader = torch.utils.data.DataLoader(
        dataset=img_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.cores
    )
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

def build_preprocessing_steps_from_args(args):
    steps = []
    print('Preprocessing steps: ', args.pp)
    for name in args.pp:
        if name == 'torso':
            steps.append(TorsoRoiPreprocessor(target_ratio=args.torso_target_ratio))
        elif name == 'square':
            steps.append(SquareImagePreprocessor(resize_size=args.square_resize,
                                                 crop_size=args.square_crop))
        else:
            raise ValueError(f"Unknown preprocessing step: {name}")

    # Sanity check for Jigsaw: last stage should deliver 255x255
    if 'square' in args.pp:
        if args.square_crop != 255:
            print("[Warning] For Jigsaw, square-crop=255 is common.")
        if args.square_resize < args.square_crop:
            raise ValueError("--square-resize must be >= --square-crop.")
    return steps
