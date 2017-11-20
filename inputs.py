import os
import random
import logger

from scipy.misc import imread
from scipy.io import loadmat


log = logger.create(__name__)


def load_in_memory(image_names, label_names):
    """
    Load all the training data into memory.
    :param image_names: list, list of strings giving all the data. The image name should be
    something like: /path/to/image.jpg.
    :param label_names: list, list of strings giving all the ground truth image.
    :return:
        data: list, list of tuples(image, label)
    """
    data = []
    for gt_path, img_path in zip(label_names, image_names):
        image = imread(img_path)
        label = load_gt_from_mat_file(gt_path)
        data.append((image, label))
    return data


def generate_image_and_gt_path(image_dir, gt_dir):
    """
    Generate two lists of file names to be processed.
    :param image_dir: str, directory where raw images are stored in.
    :param gt_dir: str, directory where ground truth images are stored in.
    :return:
        images: list, a list of strings.
        gts: list
    """
    images = []
    gts = []
    for image in os.listdir(image_dir):
        name = image.split('.')[0]
        gt_path = os.path.join(gt_dir, name + '.mat')
        img_path = os.path.join(image_dir, name + '.jpg')
        images.append(img_path)
        gts.append(gt_path)
    return images, gts


def load_gt_from_mat_file(file_name):
    """
    Load ground truth matrix from .mat file.
    :param file_name: str, the .mat file path
    :return: a 2-dimension numpy array
    """
    gt = loadmat(file_name)
    gt = gt['GTcls']
    return gt['Segmentation'][0, 0]


def load_data(image_dir, gt_dir):
    image_names, gt_names = generate_image_and_gt_path(image_dir, gt_dir)
    data = load_in_memory(image_names, gt_names)
    return data


def yield_one_example(data, n_loops=None, shuffle=False):
    """
    A generator function that yields one tuple(image, label) at a time.
    :param data: list, list of tuples(image, label)
    :param n_loops: int, number of loops through data, so `len(data) * n_loops` items will be yielded
    in total. If None, `n_loops` is set to 1.
    :param shuffle: bool, whether to shuffle data before every loop through `data`
    :return:
    """
    # get a copy of data since shuffle will reorder it.
    data = data.copy()
    if n_loops is None:
        n_loops = 1
    end = len(data) * n_loops
    for i in range(end):
        if i % len(data) == 0 and shuffle:
            random.shuffle(data)
        idx = i % len(data)
        yield data[idx][0], data[idx][1]
