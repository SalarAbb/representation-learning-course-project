# the script to run the supervised vae on a dataset with images (dsprites)
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from supervised_vae_methods_images import sdvae_model
from dataset_methods import data_set_model,dataset_dsprites

# get the data from text dataset
#assign points_train

#assign points_test

# assign values to dataset class the 
#dataset_text = data_set_model(x_train = points_train, x_test = points_test, options = [])
options_dataset = {}
options_dataset['supervise_what'] = 'size'
dataset_to_train = dataset_dsprites(options = options_dataset)
dataset_to_train.load_dataset()
#dataset_to_train = data_set_model(x_train = points_train, x_test = points_test, options = [])
# create gvae options
options = {}
options['dim_latent'] = 5 # dimension of latent variabels should be
options['activation_type'] = 'relu' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options['learning_rate'] = 1e-3 # what is learning rate (usually 0.001 to 0.01)
options['num_epochs'] = 300 # number of epochs
options['batch_size'] = 200 # number of batches in trainig
options['iteration_plot_show'] = 1000 # when to plot the results
options['print_info_step'] = 50 # how many steps to show info
options['sw_plot'] = 1 # switch for plotting the results within trials
options['plot_mode'] = 'save' #'save' or 'show'
options['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options['train_restore'] = 'restore' # 'train' or 'resotre'
options['dropout_rate'] = 0.3
options['save_global_directory'] = 'D:/Codes-Main-Python/cs699/Project/results_supervised_{}'.format(options_dataset['supervise_what'])
options['index_latent_supervised'] = np.arange(1) # this specifies the index in latent factors to be correlated with supervised target
capacity_schedule = np.concatenate( ( np.zeros((10))  ,  np.linspace(0,25,options['num_epochs'] - 20) , 25 * np.ones((10)) )  )
sdavemodel = sdvae_model(data_set = dataset_to_train, gamma = 5, capacity_schedule = capacity_schedule,mu = 50, options = options)
if options['train_restore'] is 'train':
    sdavemodel.build_computation_graph()
    sdavemodel.train_model()
elif options['train_restore'] is 'restore':
    sdavemodel.load_model(epoch_num = 300)
    sup_train_classes = dataset_to_train.y_train[:100]
    sup_train_images = dataset_to_train.x_train[:100,:,:]
    
    model_factors_mu, model_factors_log_sigma2 = sdavemodel.project_to_latent(sup_train_images)
    model_factors_mu_test = np.copy(model_factors_mu)
    model_factors_mu_test[:,0] = model_factors_mu_test[:,0] 
    sdavemodel.plot_images_from_latent(sup_train_images[0:20,:,:],model_factors_mu_test[0:20,:])
    sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[7,:,:],sup_train_images[6,:,:])

    a = 1
    #sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[9,:,:],sup_train_images[17,:,:])
    #sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[394,:,:],sup_train_images[52,:,:])
    #sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[208,:,:],sup_train_images[203,:,:])
    #sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[342,:,:],sup_train_images[354,:,:])
    #sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[340,:,:],sup_train_images[314,:,:])
    #sdavemodel.render_images_by_mixing_latent_factors(sup_train_images[7,:,:],sup_train_images[84,:,:])