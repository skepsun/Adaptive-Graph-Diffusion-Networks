import logging
import os.path

def get_logger(root):
    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(root)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger