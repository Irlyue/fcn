import tensorflow as tf
import fcn8_vgg
import fcn16_vgg
import fcn32_vgg
import utils
import logging as log


def _get_model(type_, path):
    if type_ == 'fcn32':
        return fcn32_vgg.FCN32VGG(path)
    elif type_ == 'fcn16':
        return fcn16_vgg.FCN16VGG(path)
    elif type_ == 'fcn8':
        return fcn8_vgg.FCN8VGG(path)
    else:
        raise NotImplemented


class FCN:
    def __init__(self, input_fn, n_classes, lr=1e-4, path=None, type_='fcn32', reg=None,
                 global_step=None, train=True):
        """
        :param input_fn: () -> Tensor, Tensor. function to generate images and labels tensor.
        :param n_classes: int, number of classes
        :param lr: float, learning rate
        :param path: str, path to the pretrained VGG16 model
        :param type_: str, which kind of architecture to use(fcn32, fcn16, fcn8)
        :param reg: float, regularization strength. If none, ignore weight decay.
        :param global_step: Tensor, global step
        :param train: bool, just a training flag
        """
        images, labels = input_fn()
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.global_step = global_step
        self.fcn = _get_model(type_, path)
        # first construct the FCN graph
        self.fcn.build(self.images, train=train, num_classes=n_classes)
        self.loss_op = self.loss(reg)
        self.train_op = self.train(lr)

    def train(self, lr):
        log.info('Building train operation...')
        log.info('Adding loss summary...')
        loss_averages_op = utils.add_loss_summary(self.loss_op)
        with tf.control_dependencies([loss_averages_op]):
            solver = tf.train.AdamOptimizer(lr)
            grads = solver.compute_gradients(self.loss_op)
        apply_grad_op = solver.apply_gradients(grads, global_step=self.global_step)
        log.info('Adding variable histogram...')
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        log.info('Adding gradients histogram...')
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        with tf.control_dependencies([apply_grad_op]):
            train_op = tf.no_op(name='train')
            log.info('Train operation built.')
            return train_op

    def loss(self, reg=None):
        """
        :param logits: Tensor, [None, height, width, n_classes]
        :param labels: Tensor, [None, height, width, n_classes]
        :param n_classes: int, number of classes
        :param reg: float, optional, regularization strength. If none, ignore weight decay.
        :return:
            loss: Tensor, specifying the loss
        """
        with tf.name_scope('loss'):
            logits = tf.reshape(self.fcn.upscore32, (-1, self.n_classes))
            labels = tf.reshape(self.labels, shape=(-1,))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(cross_entropy)
            if reg is not None:
                reg = tf.constant(reg, dtype=tf.float32, name='reg')
                loss += reg * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return loss
