#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import random
import numpy as np
import pdb
import math

class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.global_dis, self.local_dis = kwargs.pop('models')
        self.filter = kwargs.pop('filter')
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

    def loss_gen(self, gen, global_y_fake, local_y_fake, x_concat, x_fake, xp, pos_x, pos_y):

        # 顔の部分を覆うためのmaskを生成
        mask = Variable(xp.zeros((x_fake.shape[0], x_fake.shape[1], 64, 64), dtype=xp.float32))
        mask = F.pad(mask, ((0, 0), (0, 0), (pos_x, 192 - pos_x),(pos_y, 192 - pos_y)), "constant", constant_values=1)
        origin = x_concat[:, :3, :, :]
        #origin = origin * mask
        #generated = x_fake * mask
        origin = origin
        generated = x_fake

        x0, y0 = pos_x+32, pos_y+32
        gauss_filter = self.filter[256-x0:512-x0, 256-y0:512-y0]
        gauss_filter = chainer.cuda.to_gpu(gauss_filter)
        gauss_filter = Variable(gauss_filter)[xp.newaxis, xp.newaxis]
        gauss_filter = F.broadcast_to(gauss_filter, origin.shape)

        #再構成誤差
        #reconstruction_loss = F.sum((origin - generated) ** 2, axis=(1, 2, 3)) / (x_fake[0, :, :, :].size)
        reconstruction_loss = F.sum(((origin - generated) ** 2) * gauss_filter, axis=(1, 2, 3)) / (x_fake[0, :, :, :].size)

        batchsize = len(global_y_fake)
        loss = F.sum(F.softplus(- global_y_fake) + F.softplus(- local_y_fake) + 500 * reconstruction_loss[:, xp.newaxis]) / batchsize
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

        pos_x = np.random.randint(192) # 30
        pos_y = np.random.randint(192) # 30
        x_patch_expanded = F.pad(x_patch, ((0, 0), (0, 0), (pos_x, 192 - pos_x),(pos_y, 192 - pos_y)), "constant", constant_values=0.)

        x_concat = F.concat((x_real, x_patch_expanded), axis=1)

        xp = chainer.cuda.get_array_module(x_real.data)

        gen, global_dis, local_dis = self.gen, self.global_dis, self.local_dis
        batchsize = len(batch)

        # generatorがfake画像を生成
        x_fake = gen(x_concat)

        # global discriminator
        global_y_real = global_dis(x_real)
        global_y_fake = global_dis(x_fake)

        # local discriminator
        local_y_real = local_dis(x_patch)
        local_y_fake = local_dis(x_fake[:, :, pos_x:pos_x+64, pos_y:pos_y+64])

        global_dis_optimizer.update(self.loss_global_dis, global_dis, global_y_fake, global_y_real)
        local_dis_optimizer.update(self.loss_local_dis, local_dis, local_y_fake, local_y_real)
        gen_optimizer.update(self.loss_gen, gen, global_y_fake, local_y_fake, x_concat, x_fake, xp, pos_x, pos_y)
