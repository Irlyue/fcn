import json
import inputs
import fcn
import tensorflow as tf


FLAGS = json.load(open('config.json', 'r'))


def input_fn():
    images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
    return images, labels


with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    net = fcn.FCN(input_fn,
                  n_classes=FLAGS['n_classes'],
                  lr=FLAGS['learning_rate'],
                  path=FLAGS['vgg16_model_path'],
                  type_='fcn8',
                  reg=FLAGS['reg'],
                  global_step=global_step)

    data = inputs.load_data(FLAGS['data_image_path'], FLAGS['data_label_path'])
    with tf.Session() as sess:
        idx = 0
        image, label = data[idx]
        for i in range(50):
            # span dim
            image = image[None, :, :, :]
            feed_dict = {net.images: image, net.labels: label}
            _, loss_val = sess.run([net.train_op, net.loss_op], feed_dict)
            print('step %d, loss %.3f' % (i, loss_val))
