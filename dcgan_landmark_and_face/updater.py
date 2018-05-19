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

    def loss_global_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_local_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, global_y_fake, local_y_fake, x_concat, x_fake, xp):
        mask = Variable(xp.zeros((x_fake.shape[0], x_fake.shape[1], 64, 64), dtype=xp.float32))
        mask = F.pad(mask, ((0, 0), (0, 0), (30, 162),(30, 162)), "constant", constant_values=1)
        origin = x_concat[:, :3, :, :]
        origin = origin * mask
        generated = x_fake * mask
        reconstruction_loss = F.sum((origin - generated) ** 2, axis=(1, 2, 3)) / (x_fake[0, :, :, :].size)
        batchsize = len(global_y_fake)
        #pdb.set_trace()
        loss = F.sum(F.softplus(- global_y_fake) + F.softplus(- local_y_fake) + 500 * reconstruction_loss[:, xp.newaxis]) / batchsize
        #loss = F.sum(reconstruction_loss[:, xp.newaxis]) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        global_dis_optimizer = self.get_optimizer('global_dis')
        local_dis_optimizer = self.get_optimizer('local_dis')

        batch = self.get_iterator('main').next()
        patch = self.get_iterator('patch').next()

        x_real = Variable(self.converter(batch, self.device)) / 255.
        x_patch = Variable(self.converter(patch, self.device)) / 255.

        x_concat = F.concat((x_real, x_patch), axis=1)

        xp = chainer.cuda.get_array_module(x_real.data)

        gen, global_dis, local_dis = self.gen, self.global_dis, self.local_dis
        batchsize = len(batch)

        # generatorがfake画像を生成
        x_fake = gen(x_concat)

        # global discriminator
        global_y_real = global_dis(x_real)
        global_y_fake = global_dis(x_fake)

        # local discriminator
        local_y_real = local_dis(x_patch[:, :, 30:94, 30:94])
        local_y_fake = local_dis(x_fake[:, :, 30:94, 30:94])

        global_dis_optimizer.update(self.loss_global_dis, global_dis, global_y_fake, global_y_real)
        local_dis_optimizer.update(self.loss_local_dis, local_dis, local_y_fake, local_y_real)
        gen_optimizer.update(self.loss_gen, gen, global_y_fake, local_y_fake, x_concat, x_fake, xp)
