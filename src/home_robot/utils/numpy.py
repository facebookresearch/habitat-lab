import numpy as np
from datetime import datetime


def to_npy_file(description="data", **data):
    """write all these different data into a file. will be of the structure:
    description_datetime.npy
    """
    now = datetime.now()
    datestr = now.strftime("%Y_%m_%d-%H_%M_%S")
    filename = "%s_%s.npy" % (description, datestr)

    # This is for
    np.save(filename, data)


def load_npy_file(filename):
    """assume top level is a dictionary"""
    return np.load(filename, allow_pickle=True)[()]
