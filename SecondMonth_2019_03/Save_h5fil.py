import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


# 打开文件
f = h5py.File('ASCAD.h5', 'r')