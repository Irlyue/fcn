#!/usr/bin/env python

import scipy as scp
import scipy.misc

import logging
import tensorflow as tf
import sys

import fcn8_vgg
import utils
import os

ROOT_DIR = os.path.dirname(fcn8_vgg.__file__)
TEST_IMAGE = os.path.join(ROOT_DIR, 'test_data/tabby_cat.png')
MODEL_PATH = os.path.join(ROOT_DIR, 'pretrained_model/vgg16.npy')
print(TEST_IMAGE)
print(MODEL_PATH)

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

img1 = scp.misc.imread(TEST_IMAGE)

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=MODEL_PATH)
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variables.")

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    scp.misc.imsave(os.path.join(ROOT_DIR, './test_output/fcn8_downsampled.png'), down_color)
    scp.misc.imsave(os.path.join(ROOT_DIR, './test_output/fcn8_upsampled.png'), up_color)
