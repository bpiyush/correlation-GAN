import logging

class Logger(object):
    """docstring for Logger"""
    def __init__(self, verbose=False):
        super(Logger, self).__init__()
        if verbose:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO
        logging.basicConfig(format='=> %(asctime)s : %(levelname)s : %(message)s', datefmt='%I:%M %p', level=logging_level)

    def log(self, info):
        logging.info(info)

if __name__ == '__main__':
    logger = Logger()
    sample_info = "Testing sample info ..."
    logger.log(sample_info)
    import ipdb; ipdb.set_trace()
        