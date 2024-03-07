import logging

def get_logger():
    logger = logging.getLogger('ica')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s(%(name)s) %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
      
    logger.addHandler(ch)
    
    return logger
