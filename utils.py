import tensorflow as tf


def color_image(image, num_classes=20):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))


def add_loss_summary(loss):
    """
    :param loss: Tensor, total loss.
    :return:
        loss_average_op: operation
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = [loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_average_op = loss_averages.apply(losses)
    for item in losses:
        tf.summary.scalar('loss-raw', item)
        tf.summary.scalar('loss-smooth', loss_averages.average(item))
    return loss_average_op
