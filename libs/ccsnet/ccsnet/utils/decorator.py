import time 
import logging


def timer(func):

    def wapper(*args, **kwargs):

        start = time.time()
        func(*args, **kwargs)
        time_spent = time.time() - start

        logging.info(f"{func.__name__} spent {time_spent:.04f} sceonds")

    return wapper
