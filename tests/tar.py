import tarfile
from io import BytesIO

import numpy as np


class Tar: 

    def __init__(self, filename, mode = 'r'):
        self.filename = filename
        self.mode = mode
    
    def __enter__(self):
        self.tar = tarfile.open(self.filename, self.mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tar.close()
        return False
    
    def add(self, name, entry, save_func):
        if self.mode != 'w':
            raise ValueError('Trying to write while not in write mode')
        with BytesIO() as b:
            save_func(entry, b)
            b.seek(0)
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(b.getbuffer())
            self.tar.addfile(tarinfo, fileobj=b)

    def getnames(self):
        if self.mode != 'r':
            raise ValueError('Trying to read while not in read mode')
        return self.tar.getnames()

    def extract(self, name, load_func):
        if self.mode != 'r':
            raise ValueError('Trying to read while not in read mode')
        data = self.tar.extractfile(name)
        return load_func(data)


# from https://github.com/numpy/numpy/issues/7989#issuecomment-340921579
def np_load(data):
    array_file = BytesIO()
    array_file.write(data.read())
    array_file.seek(0)
    return np.load(array_file)