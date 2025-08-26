# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017
@author: Biagio Brattoli
Modified by: Paula Schreiber
"""
import argparse
import datetime
import os
from time import time

import numpy as np
import torch
from torch import nn

from ssl.pretext.jigsaw_absolute_position.JigsawAbsPosImageLoader import DataLoader
from ssl.pretext.jigsaw_absolute_position.JigsawAbsPosNetwork import JigsawAbsPosNetwork
from ssl.pretext.jigsaw_permutation.utils.TrainingUtils import adjust_learning_rate, configure_device, \
    load_data, log_print, load_network, save_final_model, plot_train_and_val_metrics

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=9, type=int, help='Number of patches to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=64, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()


def main():
    total_start_time = time()
    log_print('Start training: %s' % datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S'), log_dir=args.checkpoint)

    device = configure_device(args)
    log_print('Process number: %d' % (os.getpid()), log_dir=args.checkpoint)
    os.makedirs(args.checkpoint, exist_ok=True)

    train_data, train_loader = load_data(args, jigsaw_dataloader=DataLoader, train=True)
    val_data, val_loader = load_data(args, jigsaw_dataloader=DataLoader, train=False)

    iter_per_epoch = train_data.N / args.batch
    log_print('Images: train %d, validation %d' % (train_data.N, val_data.N), log_dir=args.checkpoint)

    net = initialize_network(device)
    load_network(args, net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger_test = None

    ############## TESTING ###############
    if args.evaluate:
        test(net, criterion, None, val_loader, 0, device)
        return

    ############## TRAINING ###############
    log_print(('Start training: lr %f, batch size %d, classes %d' % (args.lr, args.batch, args.classes)),log_dir=args.checkpoint)
    log_print(('Checkpoint: ' + args.checkpoint), log_dir=args.checkpoint)

    # Train the Model
    steps, train_accs, train_losses, val_accs, val_losses = train_model(criterion, device, iter_per_epoch, logger_test,net, optimizer, train_loader, val_loader)
    save_final_model(args, net)

    # Print total training time
    total_end_time = time()
    log_print('End training: %s' % datetime.datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S'),log_dir=args.checkpoint)
    duration_seconds = total_end_time - total_start_time
    m, s = divmod(duration_seconds, 60)
    h, m = divmod(m, 60)
    log_print("Training duration: %d hours, %d minutes und %d seconds" % (h, m, s), log_dir=args.checkpoint)

    plot_train_and_val_metrics(args, train_accs, train_losses, val_accs, val_losses)

    evaluate_best_model(criterion, device, net, steps, val_loader)


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

            images = images.to(device)
            # Labels are already a tensor, just move to device
            labels = labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time() - t)
            if len(net_time) > 100:
                del net_time[0]

            # We need to select the labels for the peripheral tiles, excluding the center one.
            middle_idx = int(args.classes / 2)
            peripheral_indices = [i for i in range(args.classes) if i != middle_idx]
            peripheral_labels = labels[:, peripheral_indices]

            # The output shape is (B, T-1, P). We reshape it to (B*(T-1), P) for the loss function.
            outputs_view = outputs.view(-1, args.classes)
            labels_view = peripheral_labels.view(-1)

            loss = criterion(outputs_view, labels_view)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.cpu().data.numpy())

            # Calculate accuracy
            _, predicted = torch.max(outputs_view.data, 1)
            correct = (predicted == labels_view.data).sum().item()
            acc = 100.0 * correct / labels_view.size(0)

            epoch_train_loss += loss_val
            epoch_train_acc += acc
            num_batches += 1

            if steps % 20 == 0:
                print(
                    ('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' % (
                        epoch + 1, args.epochs, steps,
                        np.mean(batch_time), np.mean(net_time),
                        lr, loss_val, acc)))

            if steps % 20 == 0:
                # logger.scalar_summary('accuracy', acc, steps)
                # logger.scalar_summary('loss', loss, steps)

                original = [im[0] for im in original]
                imgs = np.zeros([args.classes, 75, 75, 3])
                for ti, img in enumerate(original):
                    img = img.numpy()
                    imgs[ti] = np.stack([
                        (im - im.min()) / (im.max() - im.min()) if im.max() > im.min() else np.zeros_like(im)
                        for im in img
                    ], axis=2)

                # logger.image_summary('input', imgs, steps)

            steps += 1

            if steps % 1000 == 0:
                filename = '%s/jps_%03i_%06d.pth.tar' % (args.checkpoint, epoch, steps)
                net.save(filename)
                print('Saved: ' + args.checkpoint)

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
            print(f"New best model saved to {best_model_path} with validation accuracy: {val_acc:.2f}%")

        if os.path.exists(args.checkpoint + '/stop.txt'):
            # break without using CTRL+C
            break
    return steps, train_accs, train_losses, val_accs, val_losses

def test(net, criterion, logger, val_loader, steps, device):
    print('Evaluating network.......')
    accuracy = []
    losses = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = images.to(device)
        # Labels are already a tensor, just move to device
        labels = labels.to(device)

        # Forward
        outputs = net(images)

        # Reshape for loss and accuracy calculation
        # The network now outputs predictions for the peripheral tiles.
        # Output shape: (B, T-1, P)
        # Labels shape: (B, P)
        middle_idx = int(args.classes / 2)
        peripheral_indices = [i for i in range(args.classes) if i != middle_idx]
        peripheral_labels = labels[:, peripheral_indices]

        # The output shape is (B, T-1, P). We reshape it to (B*(T-1), P) for the loss function.
        outputs_view = outputs.view(-1, args.classes)
        labels_view = peripheral_labels.view(-1)

        loss = criterion(outputs_view, labels_view)
        losses.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(outputs_view.data, 1)
        correct = (predicted == labels_view.data).sum().item()
        acc = 100.0 * correct / labels_view.size(0)
        accuracy.append(acc)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracy)

    if logger is not None:
        logger.scalar_summary('accuracy', avg_acc, steps)
    print('TESTING: %d), Loss: %.3f, Accuracy %.2f%%' % (steps, avg_loss, avg_acc))
    net.train()
    return avg_loss, avg_acc

def evaluate_best_model(criterion, device, net, steps, val_loader):
    log_print("\nLoading best model for final evaluation...", log_dir=args.checkpoint)
    best_model_path = os.path.join(args.checkpoint, 'best_model.pth.tar')
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path))
        log_print("Best model loaded. Evaluating on validation set...", log_dir=args.checkpoint)
        final_loss, final_acc = test(net, criterion, None, val_loader, steps, device)
        log_print(f"Best model performance - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%", log_dir=args.checkpoint)
    else:
        log_print("No best model found to evaluate.", log_dir=args.checkpoint)

def initialize_network(device):
    net = JigsawAbsPosNetwork(num_positions=args.classes)
    net.to(device)
    log_print("Network has been moved to the selected device.", log_dir=args.checkpoint)
    return net



if __name__ == "__main__":
    main()
