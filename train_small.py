import json
import time
import inputs
import fcn
import numpy as np
import tensorflow as tf


FLAGS = json.load(open('config.json', 'r'))


def input_fn():
    images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
    return images, labels


with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    net = fcn.FCN(input_fn,
                  n_classes=FLAGS['n_classes'],
                  lr=FLAGS['learning_rate'],
                  path=FLAGS['vgg16_model_path'],
                  type_='fcn8',
                  reg=FLAGS['reg'],
                  global_step=global_step)

    data = inputs.load_data(FLAGS['data_image_path'], FLAGS['data_label_path'])
    idx = 0
    image, label = data[idx]
    image = image[None, :, :, :]
    label = label.astype(np.int64)
    correct = tf.equal(label[None, :, :], net.fcn.pred_up)
    accuracy_op = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(50):
            # span dim
            tic = time.time()
            feed_dict = {net.images: image, net.labels: label}
            _, loss_val, accuracy = sess.run([net.train_op, net.loss_op, accuracy_op], feed_dict)
            toc = time.time()
            print('step %d, loss %.3f, accuracy %.3f, time eclipsed %.3fs' % (i, loss_val, accuracy, toc - tic))

