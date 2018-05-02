#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from net import Discriminator
from net import Generator
from updater import DCGANUpdater
from visualize import out_generated_image
import pdb
import numpy as np
import cv2
import cProfile

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.ImageDataset(path, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        image = image.transpose((1, 2, 0))
        image = cv2.resize(image, (32, 32))
        image = image.transpose((2, 0, 1)).astype(np.float32)
        return image


def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-i', default='/data/unagi0/kanayama/dataset/landmark/train_small.txt',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='/data/unagi0/kanayama/dataset/dcgan-cifar/result4',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--root', '-R', default='/data/unagi0/kanayama/dataset/landmark/train/',
                        help='Root directory path of image files')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    """
    if args.dataset == '':
        # Load the CIFAR10 dataset if args.dataset is not specified
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
        train = train[:64]
        #pdb.set_trace()

    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'
              .format(args.dataset, len(image_files)))
        train = chainer.datasets\
            .ImageDataset(paths=image_files, root=args.dataset)
    """

    train = PreprocessedDataset(args.train, args.root)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
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
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, args.out),
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
