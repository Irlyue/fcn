import time
import os
import argparse
import pickle
import utils
import fcn
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_softmax
from matplotlib import colors
from tqdm import tqdm


FLAGS = utils.load_configure()
parser = argparse.ArgumentParser()
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--idx', type=int, default=0,
                    help='Which image to inference.')
parser.add_argument('--save_path', type=str, default='../others/result/')
parser.add_argument('--all', type=lambda x: x.lower() == 'true', default=False)


def image2prob(image):
    """
    Restore trained model and forward one image to get the map of probabilities.
    :param image: np.array, raw image with shape(h, w, 3)
    :return:
        prob: np.array, same shape as `image`.
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
        # apply softmax to the score
        op = tf.nn.softmax(net.fcn.upscore32, dim=3)
        with tf.Session() as sess:
            utils.restore_model(sess, FLAGS['train_dir'])
            prob = sess.run(op, feed_dict={net.images: image[None, :, :, :]})[0]
            return prob


def crf_post_process(image, prob, n_steps=5):
    """
    Use CRF as a post processing technique.
    :param image: np.array, the raw image with shape like(height, width, n_classes)
    :param prob: np.array, same shape as `image`, giving the probabilities
    :param n_steps: int, number of iterations for CRF inference.
    :return:
    """
    height, width, n_classes = prob.shape
    d = dcrf.DenseCRF2D(height, width, n_classes)

    # unary potential
    unary = unary_from_softmax(prob.transpose((2, 0, 1)))
    d.setUnaryEnergy(unary)

    # pairwise potential
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    # inference
    Q = d.inference(n_steps)
    result = np.argmax(Q, axis=0).reshape((height, width))
    return result


def main():
    tic = time.time()
    with open('result.pkl', 'rb') as f:
        data = pickle.load(f)
    idx = args.idx
    n_steps = args.n_steps
    image, label, cnn_result, _ = data[idx]
    print('CRF processing...')
    crf_result = crf_post_process(image, image2prob(image), n_steps)

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5, .5, .5, 1.0)
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, 20, 21)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.subplot(221)
    plt.imshow(image)
    plt.title('Raw Image')
    plt.subplot(222)
    plt.imshow(label, cmap=cmap, norm=norm)
    plt.title('Ground Truth')
    plt.subplot(223)
    plt.imshow(cnn_result[0], cmap=cmap, norm=norm)
    plt.title('CNN Result')
    plt.subplot(224)
    plt.imshow(crf_result, cmap=cmap, norm=norm)
    plt.title('CRF Result')
    plt.savefig('../others/result.png')
    toc = time.time()
    print('Done in %.3fs' % (toc - tic,))


def process_all_image():
    with open('result.pkl', 'rb') as f:
        data = pickle.load(f)
    n_steps = args.n_steps
    results = []
    global_step = tf.train.get_or_create_global_step()
    net = fcn.FCN(fcn.input_fn,
                  n_classes=FLAGS['n_classes'],
                  lr=FLAGS['learning_rate'],
                  path=FLAGS['vgg16_model_path'],
                  type_='fcn8',
                  reg=FLAGS['reg'],
                  global_step=global_step,
                  train=False)
    # apply softmax to the score
    op = tf.nn.softmax(net.fcn.upscore32, dim=3)
    with tf.Session() as sess:
        utils.restore_model(sess, FLAGS['train_dir'])
        for i in tqdm(range(len(data))):
            image, label, cnn_result, accuracy = data[i]
            prob = sess.run(op, feed_dict={net.images: image[None, :, :, :]})[0]
            crf_result = crf_post_process(image, prob, n_steps)
            results.append((image, label, cnn_result, crf_result, accuracy))
            path = os.path.join(args.save_path, str(i) + '.png')
            save_result(image, label, cnn_result, crf_result, path=path)


def save_result(image, label, cnn_result, crf_result, path='result.png'):
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5, .5, .5, 1.0)
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, 20, 21)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.subplot(221)
    plt.imshow(image)
    plt.title('Raw Image')
    plt.subplot(222)
    plt.imshow(label, cmap=cmap, norm=norm)
    plt.title('Ground Truth')
    plt.subplot(223)
    plt.imshow(cnn_result[0], cmap=cmap, norm=norm)
    plt.title('CNN Result')
    plt.subplot(224)
    plt.imshow(crf_result, cmap=cmap, norm=norm)
    plt.title('CRF Result')
    plt.savefig(path)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.all:
        process_all_image()
    else:
        main()
