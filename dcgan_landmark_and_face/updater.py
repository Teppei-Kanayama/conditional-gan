#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import random
import pdb

class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.global_dis, self.local_dis = kwargs.pop('models')
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
        global_dis_optimizer = self.get_optimizer('global_dis')
        local_dis_optimizer = self.get_optimizer('local_dis')

        batch = self.get_iterator('main').next()
        patch = self.get_iterator('patch').next()
        #pdb.set_trace()

        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, global_dis, local_dis = self.gen, self.global_dis, self.local_dis
        batchsize = len(batch)

        y_real = global_dis(x_real)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fake = global_dis(x_fake)

        #rand_num = random.randint(0, 1)
        #if rand_num == 0:
        global_dis_optimizer.update(self.loss_dis, global_dis, y_fake, y_real)
        #local_dis_optimizer.update()
        gen_optimizer.update(self.loss_gen, gen, y_fake)
