#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F

import cv2
import pdb

from net import Generator
import argparse

def make_image(landmark_path, face_path, gen, pos_x, pos_y):
    xp = gen.xp

    image = Image.open(landmark_path).resize((256, 256))
    patch = Image.open(face_path).resize((64, 64))

    image = xp.asarray(image).astype(xp.float32)
    patch = xp.asarray(patch).astype(xp.float32)

    #pdb.set_trace()
    image = xp.transpose(image, (2, 0, 1))
    patch = xp.transpose(patch, (2, 0, 1))

    image = Variable(image) / 255.
    patch = Variable(patch) / 255.

    patch_expanded = F.pad(patch, ((0, 0), (pos_x, 192 - pos_x),(pos_y, 192 - pos_y)), "constant", constant_values=0.)

    concat_image = F.concat((image, patch_expanded), axis=0)
    concat_image = F.expand_dims(concat_image, axis=0)

    with chainer.using_config('train', False):
        x = gen(concat_image)
    x = chainer.cuda.to_cpu(x.data)

    #pdb.set_trace()

    img = x[0].transpose(1, 2, 0) * 255.
    img = img.astype(np.uint8)
    img_with_bbox = cv2.rectangle(img.copy(),(pos_y, pos_x),(pos_y+64,pos_x+64),(0,255,0),3)

    img = Image.fromarray(img)
    img_with_bbox = Image.fromarray(img_with_bbox)

    img.save("./samples/dst4.png")
    img_with_bbox.save("./samples/bbox.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    args = parser.parse_args()

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden)
    chainer.serializers.load_npz("/data/unagi0/kanayama/dataset/landmark/results/result13/gen_iter_92000.npz", gen)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU

    # 0 ~ 192
    pos_x = 120
    pos_y = 180

    make_image("./samples/hachiko.jpg", "./samples/myface.jpg", gen, pos_x, pos_y)
