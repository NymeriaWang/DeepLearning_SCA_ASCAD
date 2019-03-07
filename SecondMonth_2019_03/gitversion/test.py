import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from keras.models import load_model
from scipy import io


#-------------------------------------------------------

Y_profiling = scio.loadmat('TrainLabels.mat')
a=type(Y_profiling)
#Y_profiling = Y_profiling['TrainLabels']

#b = a[:,320:370]
#print(Y_profiling.shape)
#print(Y_profiling)
print(a)
#----------------------------------------------------------------------


