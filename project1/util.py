import time
from functools import wraps

class timer(object):
    def __init__(self, info="none"):
        self.info = info

    def __call__(self, func):
        @wraps(func)
        def wrapper(*arg, **kwargs):
            t0 = time.time()
            ret = func(*arg, **kwargs)
            print("{} time: {}".format(self.info, time.time()-t0))
            return ret
        return wrapper