# the script to run the supervised vae on a dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from supervised_vae_methods import supervised_vae_model
from dataset_methods import data_set_model 

# get the data from text dataset
#assign points_train

#assign points_test

# assign values to dataset class the 
dataset_text = data_set_model(x_train = points_train, x_test = points_test, options = [])
# create bvae options
options = {}
options['dim_latent'] = dataset_text.dim_x # dimension of latent variabels should be
options['activation_type'] = 'relu' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options['learning_rate'] = 1e-3 # what is learning rate (usually 0.001 to 0.01)
options['num_epochs'] = 400 # number of epochs
options['batch_size'] = 400 # number of batches in trainig
options['iteration_plot_show'] = 1000 # when to plot the results
options['print_info_step'] = 50 # how many steps to show info
options['sw_plot'] = 1 # switch for plotting the results within trials
options['plot_mode'] = 'save' #'save' or 'show'
options['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options['train_restore'] = 'restore' # 'train' or 'resotre'
options['dropout_rate'] = 0.3
options['save_global_directory'] = 'D:/Codes-Main-Python/cs699/Project/results'
options['index_latent_supervised'] = np.arange(3) # this specifies the index in latent factors to be correlated with supervised target

bvaemodel = supervised_vae_model(data_set = dataset_text, beta = 0, mu = 0.1, options = options)
if options['train_restore'] is 'train':
    bvaemodel.build_computation_graph()
    bvaemodel.train_model()
elif options['train_restore'] is 'restore':
    bvaemodel.load_model(epoch_num = 400)
