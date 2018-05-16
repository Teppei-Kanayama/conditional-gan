#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import pdb


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            #self.c0 = L.Convolution2D(6, ch, 3, 1, 1, initialW=w)
            self.c6_0 =  L.Convolution2D(6, 16, 3, 1, 1, initialW=w)
            self.c6_1 =  L.Convolution2D(16, 16, 3, 1, 1, initialW=w)
            self.c5_0 =  L.Convolution2D(16, 32, 3, 1, 1, initialW=w)
            self.c5_1 =  L.Convolution2D(32, 32, 3, 1, 1, initialW=w)
            self.c4_0 =  L.Convolution2D(32, 64, 3, 1, 1, initialW=w)
            self.c4_1 =  L.Convolution2D(64, 64, 3, 1, 1, initialW=w)
            self.c3_0 =  L.Convolution2D(64, 128, 3, 1, 1, initialW=w)
            self.c3_1 =  L.Convolution2D(128, 128, 3, 1, 1, initialW=w)
            self.c2_0 =  L.Convolution2D(128, 256, 3, 1, 1, initialW=w)
            self.c2_1 =  L.Convolution2D(256, 256, 3, 1, 1, initialW=w)
            self.c1_0 =  L.Convolution2D(256, 512, 3, 1, 1, initialW=w)
            self.c1_1 =  L.Convolution2D(512, 512, 3, 1, 1, initialW=w)

            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2 * 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4 * 2, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8 * 2, ch // 16, 4, 2, 1, initialW=w)
            self.dc5 = L.Deconvolution2D(ch // 16 * 2, ch // 32, 4, 2, 1, initialW=w)
            self.dc6 = L.Deconvolution2D(ch // 32 * 2, 3, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)
            self.bn4 = L.BatchNormalization(ch // 16)
            self.bn5 = L.BatchNormalization(ch // 32)

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, x):
        
        g5 = F.max_pooling_2d(F.relu(self.c6_1(F.dropout(F.relu(self.c6_0(x)), 0.2))), 2)
        g4 = F.max_pooling_2d(F.relu(self.c5_1(F.dropout(F.relu(self.c5_0(g5)), 0.2))), 2)
        g3 = F.max_pooling_2d(F.relu(self.c4_1(F.dropout(F.relu(self.c4_0(g4)), 0.2))), 2)
        g2 = F.max_pooling_2d(F.relu(self.c3_1(F.dropout(F.relu(self.c3_0(g3)), 0.2))), 2)
        g1 = F.max_pooling_2d(F.relu(self.c2_1(F.dropout(F.relu(self.c2_0(g2)), 0.2))), 2)
        h0 = F.max_pooling_2d(F.relu(self.c1_1(F.dropout(F.relu(self.c1_0(g1)), 0.2))), 2)

        h1 = F.relu(self.bn1(self.dc1(h0)))
        i1 = F.concat((h1, g1), axis=1)

        h2 = F.relu(self.bn2(self.dc2(i1)))
        i2 = F.concat((h2, g2), axis=1)

        h3 = F.relu(self.bn3(self.dc3(i2)))
        i3 = F.concat((h3, g3), axis=1)

        h4 = F.relu(self.bn4(self.dc4(i3)))
        i4 = F.concat((h4, g4), axis=1)

        h5 = F.relu(self.bn5(self.dc5(i4)))
        i5 = F.concat((h5, g5), axis=1)

        x = F.sigmoid(self.dc6(i5))
        #pdb.set_trace()
        return x


class GlobalDiscriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(GlobalDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 2, 1, initialW=w) #
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 2, 1, initialW=w) #
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 2, 1, initialW=w) #
            self.l4 = L.Linear(bottom_width * bottom_width * ch, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = add_noise(x)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        return self.l4(h)

class LocalDiscriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(LocalDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 2, 1, initialW=w) #
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 2, 1, initialW=w) #
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 2, 1, initialW=w) #
            self.l4 = L.Linear(bottom_width * bottom_width * ch, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = add_noise(x)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        #h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        #h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        #pdb.set_trace()
        return self.l4(h)
