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


class dataset_dsprites:
    def __init__(self,options):
        ids = np.load('ids.npz')
        self.ids = ids
        self.options = options
        if options['supervise_what'] == 'shape':
            self.index_supervision = 1
        elif options['supervise_what'] == 'size':
            self.index_supervision = 2

    def load_dataset(self):
        data = np.load('data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz') # download from github
        print(data.files) #Prints: [’’imgs’, ’latents_classes’, ’latents_values’, ...]
        all_imgs = data['imgs']
        self.x_train = all_imgs[ self.ids[ 'train' ] ] # please only access imgs of ’train’
        self.x_test = all_imgs[ self.ids['test_reconstruct'] ]
        self.y_train = data['latents_classes'][ self.ids[ 'train' ] ][:,self.index_supervision]
        self.y_test = data['latents_classes'][ self.ids['test_reconstruct'] ][:,self.index_supervision]
        if self.options['supervise_what'] == 'shape':
            self.dim_y = 3
        elif self.options['supervise_what'] == 'size':
            self.dim_y = 6
        # You can access all attributes for supervised_train!
        # self.sup_train_images = all_imgs[ self.ids[ 'supervised_train' ] ]
        # self.sup_train_factors = data['latents_values'][ self.ids[ 'supervised_train' ] ]
        # self.sup_train_classess = data['latents_classes'][self.ids['supervised_train'] ]
        # self.test_reconstruct = all_imgs[ self.ids['test_reconstruct'] ] # for q1_a.pdf
        # self.test_interpolate = all_imgs[ self.ids['test_interpolate'] ] # for q1_b.pdf


