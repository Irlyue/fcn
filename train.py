import os
import json
import fcn
import time
import utils
import logger
import argparse

import inputs
import tensorflow as tf

# load the configuration file
FLAGS = json.load(open('config.json'))
log = logger.create(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--recover', type=lambda x: x.lower() != "false", default=True,
                    help='Whether to recover from last run.')


def train(recover):
    """
    :param recover: bool, whether to recover from last run.
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
                      global_step=global_step)

        data = inputs.load_data(FLAGS['data_image_path'], FLAGS['data_label_path'])
        log.info('Data loaded successfully, %d items in total!' % len(data))

        writer = tf.summary.FileWriter(FLAGS['train_dir'])
        writer.add_graph(tf.get_default_graph())
        saver = tf.train.Saver()

        with tf.Session() as sess:
            base_step = 0
            if recover:
                log.info('Restoring model from last run...')
                tic = time.time()
                ckpt = tf.train.get_checkpoint_state(FLAGS['train_dir'])
                if ckpt and ckpt.model_checkpoint_path:
                    base_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    log.info('Successfully loaded model-%d' % base_step)
                else:
                    log.warning('No checkpoint file found!')
                toc = time.time()
                log.info('Done in %.3fs' % (toc - tic,))
            else:
                tf.global_variables_initializer().run()
            log.info('Begin training...')
            for i, (image, label) in enumerate(inputs.yield_one_example(data, FLAGS['n_loops'], True)):
                step = i + base_step
                # span dim
                image = image[None, :, :, :]
                feed_dict = {net.images: image, net.labels: label}
                sess.run(net.train_op, feed_dict)
                if step % FLAGS['print_every'] == 0:
                    loss_val = sess.run(net.loss_op, feed_dict)
                    print('step %d, loss %.3f' % (step, loss_val))
                if step % FLAGS['save_every'] == 0:
                    log.info('Saving model in step %d...' % step)
                    tic = time.time()
                    path = os.path.join(FLAGS['train_dir'], 'model')
                    saver.save(sess, path, global_step=global_step)
                    toc = time.time()
                    log.info('Done in %.3f' % (toc - tic,))


def main(recover):
    if recover:
        train_from_recover()
    else:
        train_from_scratch()


def train_from_scratch():
    log.info(json.dumps(FLAGS, indent=2, sort_keys=True))
    utils.delete_if_exists_and_create(FLAGS['train_dir'])
    utils.maybe_download(FLAGS['vgg16_url'], FLAGS['vgg16_dir'])
    train(recover=False)


def train_from_recover():
    train(recover=True)


if __name__ == '__main__':
    args = parser.parse_args()
    main(recover=args.recover)
