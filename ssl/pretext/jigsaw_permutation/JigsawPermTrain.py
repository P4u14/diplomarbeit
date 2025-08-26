# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import argparse
import datetime
import os
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable

from ssl.dataset.JigsawImageLoader import DataLoader
from ssl.pretext.jigsaw_permutation.JigsawPermNetwork import JigsawPermNetwork
from ssl.pretext.jigsaw_permutation.utils.TrainingUtils import adjust_learning_rate, compute_accuracy

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=64, type=int, help='batch size')
parser.add_argument('--checkpoint', default='data/SSL_Pretext/JigsawPerm', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()


def main():
    total_start_time = time()
    log_print('Start training: %s' % datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S'))

    device = configure_device()
    log_print('Process number: %d' % (os.getpid()))
    os.makedirs(args.checkpoint,exist_ok=True)

    train_data, train_loader = load_train_data()
    val_data, val_loader = load_val_data()

    iter_per_epoch = train_data.N / args.batch
    log_print('Images: train %d, validation %d' % (train_data.N, val_data.N))

    net = initialize_network(device)
    load_network(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger_test = None

    ############## TESTING ###############
    if args.evaluate:
        test(net, criterion, None, val_loader, 0, device)
        return

    ############## TRAINING ###############
    log_print(('Start training: lr %f, batch size %d, classes %d' % (args.lr, args.batch, args.classes)))
    log_print(('Checkpoint: ' + args.checkpoint))

    # Train the Model
    steps, train_accs, train_losses, val_accs, val_losses = train_model(criterion, device, iter_per_epoch, logger_test, net, optimizer, train_loader, val_loader)
    save_final_model(net)

    # Print total training time
    total_end_time = time()
    log_print('End training: %s' % datetime.datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S'))
    duration_seconds = total_end_time - total_start_time
    m, s = divmod(duration_seconds, 60)
    h, m = divmod(m, 60)
    log_print("Training duration: %d hours, %d minutes und %d seconds" % (h, m, s))

    plot_train_and_val_metrics(train_accs, train_losses, val_accs, val_losses)

    evaluate_best_model(criterion, device, net, steps, val_loader)


def evaluate_best_model(criterion, device, net, steps, val_loader):
    log_print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(args.checkpoint, 'best_model.pth.tar')
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path))
        log_print("Best model loaded. Evaluating on validation set...")
        final_loss, final_acc = test(net, criterion, None, val_loader, steps, device)
        log_print(f"Best model performance - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%")
    else:
        log_print("No best model found to evaluate.")


def plot_train_and_val_metrics(train_accs, train_losses, val_accs, val_losses):
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
    log_print(f"Training metrics plot saved to {plot_path}")
    plt.close()


def save_final_model(net):
    final_model_path = os.path.join(args.checkpoint, 'final_model.pth.tar')
    net.save(final_model_path)
    log_print(f"Final model saved to {final_model_path}")


def train_model(criterion, device, iter_per_epoch, logger_test, net, optimizer, train_loader, val_loader):
    batch_time, net_time = [], []
    steps = args.iter_start
    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(int(args.iter_start / iter_per_epoch), args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)

        end = time()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_batches = 0
        for i, (images, labels, original) in enumerate(train_loader):
            batch_time.append(time() - end)
            if len(batch_time) > 100:
                del batch_time[0]

            images = Variable(images)
            labels = Variable(labels)
            images = images.to(device)
            labels = labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time() - t)
            if len(net_time) > 100:
                del net_time[0]

            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            acc = prec1.item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.cpu().data.numpy())

            epoch_train_loss += loss_val
            epoch_train_acc += acc
            num_batches += 1

            if steps % 20 == 0:
                log_print(
                    ('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' % (
                        epoch + 1, args.epochs, steps,
                        np.mean(batch_time), np.mean(net_time),
                        lr, loss_val, acc)))

            if steps % 20 == 0:
                original = [im[0] for im in original]
                imgs = np.zeros([9, 75, 75, 3])
                for ti, img in enumerate(original):
                    img = img.numpy()
                    imgs[ti] = np.stack([
                        (im - im.min()) / (im.max() - im.min()) if im.max() > im.min() else np.zeros_like(im)
                        for im in img
                    ], axis=2)

            steps += 1

            if steps % 1000 == 0:
                filename = '%s/jps_%03i_%06d.pth.tar' % (args.checkpoint, epoch, steps)
                net.save(filename)
                log_print('Saved: ' + args.checkpoint)

            end = time()

        # Store average epoch metrics for training
        train_losses.append(epoch_train_loss / num_batches)
        train_accs.append(epoch_train_acc / num_batches)

        val_loss, val_acc = test(net, criterion, logger_test, val_loader, steps, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.checkpoint, 'best_model.pth.tar')
            net.save(best_model_path)
            log_print(f"New best model saved to {best_model_path} with validation accuracy: {val_acc:.2f}%")

        if os.path.exists(args.checkpoint + '/stop.txt'):
            # break without using CTRL+C
            break
    return steps, train_accs, train_losses, val_accs, val_losses


def load_val_data():
    val_path = args.data + '/ILSVRC2012_img_val'
    if os.path.exists(val_path + '_255x255'):
        val_path += '_255x255'
    val_data = DataLoader(val_path, args.data + '/ilsvrc12_val.txt', classes=args.classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                             num_workers=args.cores)
    return val_data, val_loader


def load_train_data():
    train_path = args.data + '/ILSVRC2012_img_train'
    if os.path.exists(train_path + '_255x255'):
        train_path += '_255x255'
    train_data = DataLoader(train_path, args.data + '/ilsvrc12_train.txt', classes=args.classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.cores)
    return train_data, train_loader


def initialize_network(device):
    net = JigsawPermNetwork(args.classes)
    net.to(device)
    log_print("Network has been moved to the selected device.")
    return net


def load_network(net):
    # load from checkpoint if exists, otherwise from model
    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files) > 0:
            files.sort()
            # print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint + '/' + ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            log_print('Starting from: ', ckp)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)


def configure_device():
    if args.gpu >= 0 and torch.backends.mps.is_available():
        device = torch.device("mps")
        log_print("MPS available. Training runs on MPS.")
    elif args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        log_print(('Using GPU %d' % args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        device = torch.device("cpu")
        log_print('Training läuft auf CPU.')
    return device


def test(net, criterion, logger, val_loader, steps, device):
    log_print('Evaluating network.......')
    accuracy = []
    losses = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        labels = Variable(labels)
        # if args.gpu >= 0 and torch.cuda.is_available():
        #     images = images.cuda()
        images = images.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        outputs = outputs.cpu().data
        labels = labels.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1.item())

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracy)

    if logger is not None:
        logger.scalar_summary('accuracy', avg_acc, steps)
    log_print('TESTING: %d), Loss: %.3f, Accuracy %.2f%%' % (steps, avg_loss, avg_acc))
    net.train()
    return avg_loss, avg_acc


def log_print(*text, sep=' ', end='\n', file=None, flush=False):
    # standard output
    print(*text, sep=sep, end=end, file=file, flush=flush)
    # write in log file in checkpoint folder
    log_dir = args.checkpoint
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'logs.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        print(*text, sep=sep, end=end, file=f)

if __name__ == "__main__":
    main()