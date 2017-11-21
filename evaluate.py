import time
import json
import fcn
import inputs
import logger
import tensorflow as tf
import numpy as np


FLAGS = json.load(open('config.json', 'r'))
log = logger.create(__name__)


def eval_train():
    """
    Evaluate training set.
    :return:
    """
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        net = fcn.FCN(fcn.input_fn,
                      n_classes=FLAGS['n_classes'],
                      lr=FLAGS['learning_rate'],
                      path=FLAGS['vgg16_model_path'],
                      type_='fcn8',
                      reg=FLAGS['reg'],
                      global_step=global_step,
                      train=False)

        data = inputs.load_data(FLAGS['data_image_path'], FLAGS['data_label_path'])
        log.info('Data loaded successfully, %d items in total!' % len(data))

        correct = tf.equal(net.labels, net.fcn.pred_up)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            counter = 0
            while True:
                counter += 1
                eval_once(sess, saver, correct, data, net)
                if FLAGS['run_once'] or counter > FLAGS['n_evals']:
                    break
                time.sleep(FLAGS['save_every_seconds'])


def eval_once(sess, saver, correct, data, net):
    tic = time.time()
    ckpt = tf.train.get_checkpoint_state(FLAGS['train_dir'])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        base_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        log.info('Successfully restored model-%d' % base_step)
    else:
        log.warning('No checkpoint file found!')
    total_count = 0
    true_count = 0
    for i in range(len(data)):
        image, label = data[i]
        image = image[None, :, :, :]
        label = label.astype(np.int64)
        result = sess.run(correct, feed_dict={net.images: image, net.labels: label})
        total_count += result.size
        true_count += np.sum(result)
    accuracy = true_count * 1.0 / total_count
    toc = time.time()
    log.info('Accuracy @1 = %.3f(%.3fs)' % (accuracy, toc - tic))


if __name__ == '__main__':
    eval_train()
