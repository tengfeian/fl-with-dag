import logging
import os


def set_log(log_file="default.log"):
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        filename=log_file, level=logging.INFO)
    logging.info('Started logging')