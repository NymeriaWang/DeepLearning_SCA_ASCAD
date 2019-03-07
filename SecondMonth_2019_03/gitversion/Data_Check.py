import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy import io


#------------------------read npy to mat-------------------------------


keys = np.load('keylists.npy')
io.savemat('keys.mat', {'keys': keys})

plaintexts = np.load('textin.npy')
io.savemat('plaintexts.mat', {'plaintexts': plaintexts})

ciphertexts = np.load('textout.npy')
io.savemat('ciphertexts.mat', {'ciphertexts': ciphertexts})

traces = np.load('traces.npy')
io.savemat('traces.mat', {'traces': traces})


#----------------------------------------------------------------------


