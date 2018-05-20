#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from net import GlobalDiscriminator, LocalDiscriminator
from net import Generator
from updater import DCGANUpdater
from visualize import out_generated_image
import pdb
import numpy as np
import cv2
import cProfile
import math

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.ImageDataset(path, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        image = image.transpose((1, 2, 0))
        image = cv2.resize(image, (256, 256))
        image = image.transpose((2, 0, 1)).astype(np.float32)
        return image


class PreprocessedDataset2(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.ImageDataset(path, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        image = image.transpose((1, 2, 0))
        image = cv2.resize(image, (64, 64))
        image = image.transpose((2, 0, 1)).astype(np.float32)
        #expand_image = np.zeros((3, 256, 256), dtype=np.float32)
        #expand_image[:, 30:94, 30:94] = image
        #return expand_image
        return image

def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-i', default='/data/ugui0/kanayama/landmark_large.txt',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='/data/unagi0/kanayama/dataset/landmark/results/result15',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=2000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--root', '-R', default='/data/ugui0/kanayama/train',
                        help='Root directory path of image files')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    gauss_filter = np.zeros((256 * 2, 256 * 2), dtype=np.float32)
    sigma = 50
    x0 = y0 = 256

    for x in range(256 * 2):
        for y in range(256 * 2):
            gauss_filter[x][y] = 1 - np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden)
    global_dis = GlobalDiscriminator()
    local_dis = LocalDiscriminator()

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        global_dis.to_gpu()
        local_dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_global_dis = make_optimizer(global_dis)
    opt_local_dis = make_optimizer(local_dis)

    train = PreprocessedDataset(args.train, args.root)
    patch = PreprocessedDataset2('/data/unagi0/kanayama/dataset/celeba.txt', '/data/ugui0/kanayama/celeba/')

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    patch_iter = chainer.iterators.MultiprocessIterator(patch, args.batchsize)
    #pdb.set_trace()
    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, global_dis, local_dis),
        filter = gauss_filter,
        iterator={
            'main': train_iter, 'patch': patch_iter},
        optimizer={
            'gen': opt_gen, 'global_dis': opt_global_dis, 'local_dis': opt_local_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        global_dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'global_dis/loss', 'local_dis/loss', 'gen/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, global_dis,
            10, 10, train_iter, patch_iter, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pr.print_stats()
    pr.dump_stats('fib.profile')
