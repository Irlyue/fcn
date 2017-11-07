#!/usr/bin/env python

import scipy as scp
import scipy.misc

import tensorflow as tf

import fcn32_vgg
import utils
import os

ROOT_DIR = os.path.dirname(fcn32_vgg.__file__)
TEST_IMAGE = os.path.join(ROOT_DIR, 'test_data/tabby_cat.png')
MODEL_PATH = os.path.join(ROOT_DIR, 'pretrained_model/vgg16.npy')
print(TEST_IMAGE)
print(MODEL_PATH)

img1 = scp.misc.imread(TEST_IMAGE)

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn32_vgg.FCN32VGG(vgg16_npy_path=MODEL_PATH)
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    scp.misc.imsave(os.path.join(ROOT_DIR, './test_output/fcn32_downsampled.png'), down_color)
    scp.misc.imsave(os.path.join(ROOT_DIR, './test_output/fcn32_upsampled.png'), up_color)
