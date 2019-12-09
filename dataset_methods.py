# dataset method to handle dataset manipulation
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats, io

class data_set_model(object):
    # this class creates the RNN model for movement generation
    
    def __init__(self,x_train,x_test,y_train,y_test,options):
        # options: options of loading the RNN model
        #  x and y are always time * dim
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.dim_x = np.shape(x_test)[1]
        self.dim_y = np.shape(y_test)[1]

