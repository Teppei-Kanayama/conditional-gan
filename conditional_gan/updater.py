#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable
import pdb
import random

class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        # batch: list (len=batchsize)
        # batch[i]: numpy array (shape=(channels, height, width))

        # batch: list (len=batchsize)
        # batch[i][0]: numpy array (shape=(channels, height, width))
        # batch[i][1]: int (shape=(channels, height, width))

        #pdb.set_trace()
        in_array = self.converter(batch, self.device)
        images = in_array[0]
        labels = in_array[1]

        x_real = Variable(images) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        # calculate the probability of real
        y_real = dis(x_real, labels)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))

        # create fake image
        x_fake = gen(z, labels)

        # calculate the probability of real
        y_fake = dis(x_fake, labels)

        rand_num = random.randint(0, 1)
        if rand_num == 0:
            dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
