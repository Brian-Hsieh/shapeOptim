import os
from contextlib import contextmanager

@contextmanager
def cwd(path, print_path = False):
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        if print_path:
            print('Access to location: {}'.format(path))
        os.chdir(old_path)