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

        for i in range(5):
            image = train_iter.next()[i]
            patch = patch_iter.next()[i]

            #import pdb;pdb.set_trace()
            pil_image = Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))
            pil_patch = Image.fromarray(np.transpose(patch, (1, 2, 0)).astype(np.uint8))

            image = Variable(xp.asarray(image)) / 255.
            patch = Variable(xp.asarray(patch)) / 255.

            concat_image = F.concat((image, patch), axis=0)
            concat_image = F.expand_dims(concat_image, axis=0)

            with chainer.using_config('train', False):
                x = gen(concat_image)
            x = chainer.cuda.to_cpu(x.data)

            #pdb.set_trace()

            img = x[0].transpose(1, 2, 0) * 255.
            img = img.astype(np.uint8)
            img_with_bbox = cv2.rectangle(img.copy(),(30, 30),(94,94),(0,255,0),3)

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
