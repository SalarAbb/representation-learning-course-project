# dataset method to handle dataset manipulation
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats, io
import csv


class data_set_text(object):
    
    def __init__(self,options):
        self.label_path = options['label_path']
        self.rep_path = options['rep_path']

    def load_dataset(self):    
        # read labels
        with open(self.label_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            gender_list = []
            for row in csv_reader:
                if line_count != 0:
                    gender_list.append(row[0])
                line_count += 1
                if line_count % 10000 == 0:
                    print('{}'.format(line_count/10000))
            print(f'Processed {line_count} lines.')
        gender_vec = [self._transform_gender_to_digit(x) for x in gender_list]
        gender_vec = np.asarray(gender_vec)    
        # read bert re[]
        data =  np.load(self.rep_path) 
        # index_train
        index_train = np.arange(50000)
        index_test = np.arange(60000)
        index_test = np.delete(index_test,index_train)
        #
        which_data_index = np.arange(40)
        self.y_train = gender_vec[index_train]
        self.x_train = data[index_train,:]
        self.x_train = self.x_train[:,which_data_index]
        self.y_test = gender_vec[index_test]
        self.x_test = data[index_test,:]
        self.x_test = self.x_test[:,which_data_index]
        self.dim_y = 1
        self.dim_x = np.shape(self.x_train)[1]
    def _transform_gender_to_digit(self,gender):
        if gender == 'Male':
            return 0
        elif gender == 'Female':
            return 1


class data_set_model(object):

    
    def __init__(self,x_train,x_test,y_train,y_test,options):
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


