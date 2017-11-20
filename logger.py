import logging
import sys
import json
import os
import utils


FLAGS = json.load(open('config.json', 'r'))
LOG_FILE_NAME = os.path.join(FLAGS['log_dir'], FLAGS['log_file'])
utils.create_if_not_exists(FLAGS['log_dir'])


def create(name, level=logging.INFO, handlers=('stdout', 'file')):
    """
    Create logger given some specifics.
    :param name: str, name of the logger
    :param level: int, comes from standard library `logging`
    :param handlers: list or tuple, a sequence of `str` specifying the type of handler.
    :return:
        log: logging.Logger
    """
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(level)
    for handler in handlers:
        log.addHandler(create_handler(handler))
    return log


def create_handler(handler_name):
    """
    :param handler_name: str,
    :return:
    """
    if handler_name == 'stdout':
        ans = logging.StreamHandler(sys.stdout)
    elif handler_name == 'file':
        ans = logging.FileHandler(LOG_FILE_NAME)
    else:
        raise NotImplementedError
    return ans
