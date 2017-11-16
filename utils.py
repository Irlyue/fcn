import tensorflow as tf
import sys
import os
import urllib.request


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


def delete_if_exists_and_create(folder):
    if tf.gfile.Exists(folder):
        tf.gfile.DeleteRecursively(folder)
    os.makedirs(folder)


def create_if_not_exists(folder):
    if not tf.gfile.Exists(folder):
        os.makedirs(folder)


def maybe_download(url, dest_dir):
    """
    :param url: str, URL str to download from
    :param dest_dir: str, directory to put the downloaded file.
    :return:
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print('Directory `%s` not existed and new one created!' % (dest_dir,))
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>>> Downloading %s %.1f%%' % (filename, count * block_size * 100.0 / total_size))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        file_stat = os.stat(filepath)
        print('Successfully downloaded', filename, file_stat.st_size, 'bytes!!!')
