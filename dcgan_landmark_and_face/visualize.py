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


def out_generated_image(gen, dis, rows, cols, train_iter, patch_iter, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        images = train_iter.next()
        patches = patch_iter.next()

        for i in range(5):
            image = images[i]
            patch = patches[i]

            #import pdb;pdb.set_trace()
            pil_image = Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))
            pil_patch = Image.fromarray(np.transpose(patch, (1, 2, 0)).astype(np.uint8))

            image = Variable(xp.asarray(image)) / 255.
            patch = Variable(xp.asarray(patch)) / 255.

            pos_x = np.random.randint(192) # 30
            pos_y = np.random.randint(192) # 30

            patch_expanded = F.pad(patch, ((0, 0), (pos_x, 192 - pos_x),(pos_y, 192 - pos_y)), "constant", constant_values=0.)

            concat_image = F.concat((image, patch_expanded), axis=0)
            concat_image = F.expand_dims(concat_image, axis=0)

            with chainer.using_config('train', False):
                x = gen(concat_image)
            x = chainer.cuda.to_cpu(x.data)

            #pdb.set_trace()

            img = x[0].transpose(1, 2, 0) * 255.
            img = img.astype(np.uint8)
            img_with_bbox = cv2.rectangle(img.copy(),(pos_x, pos_y),(pos_x+64,pos_y+64),(0,255,0),3)

            img = Image.fromarray(img)
            img_with_bbox = Image.fromarray(img_with_bbox)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>8}'.format(trainer.updater.iteration) + '_' + str(i) + '.png'
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)

            img.save(preview_path)
            img_with_bbox.save(preview_path + "_bbox.png")
            pil_image.save(preview_path + "_landscape.png")
            pil_patch.save(preview_path + "_face.png")

    return make_image
