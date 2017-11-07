import tensorflow as tf
import json
import fcn
import time
import sys
import logging as log

log.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                level=log.INFO,
                stream=sys.stdout)

# load the configuration file
FLAGS = json.load(open('config.json'))


def input_fn():
    images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[1, None, None, FLAGS['n_classes']], name='labels')
    return images, labels


def train(recover):
    """
    :param recover: bool, whether to recover from last run.
    :return:
    """
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        net = fcn.FCN(input_fn,
                      n_classes=FLAGS['n_classes'],
                      lr=FLAGS['learning_rate'],
                      path=FLAGS['vgg16_model_path'],
                      type_='fcn8',
                      reg=FLAGS['reg'],
                      global_step=global_step)

        class _MyHooker(tf.train.SessionRunHook):
            def __init__(self):
                pass

            def begin(self):
                self._step = -1
                self._tic = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(net.loss_op)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS['print_every'] == 0:
                    toc = time.time()
                    duration = toc - self._tic
                    self._tic = toc
                    loss_value = run_values.results
                    duration_per_step = duration / FLAGS['print_every']
                    fmt_spec = "step %d, loss %.3f, %.2fsec"
                    log.info(fmt_spec % (self._step, loss_value, duration_per_step))

        hooker = _MyHooker()
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS['train_dir'],
            save_checkpoint_secs=FLAGS['save_every_seconds'],
            hooks=[
                tf.train.NanTensorHook(net.loss_op),
                tf.train.StopAtStepHook(num_steps=FLAGS['max_steps']),
                _MyHooker()
            ]
        ) as sess:
            if recover:
                log.info('Trying to recover from last run...')
                ckpt = tf.train.get_checkpoint_state(FLAGS['train_dir'])
                if ckpt and ckpt.model_checkpoint_path:
                    step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    hooker._step = step
                    log.info('Model-%d recovered successfully!' % (step, ))
                else:
                    log.info('No checkpoint file found!')
            # while not sess.should_stop():
            #     log.info('Begin training...')
            #     sess.run(net.train_op)


def main(recover):
    if not recover:
        log.info('Training from scratch...')
        if tf.gfile.Exists(FLAGS['train_dir']):
            log.warning('"%s" exists and will be deleted!' % (FLAGS['train_dir'],))
            tf.gfile.DeleteRecursively(FLAGS['train_dir'])
    train(recover)


if __name__ == '__main__':
    main(recover=True)