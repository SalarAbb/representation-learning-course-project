# the script to run the supervised vae on a dataset with R^n data points
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from supervised_vae_methods import sdvae_model
from dataset_methods import data_set_model,dataset_dsprites,data_set_text
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
# get the data from text dataset
#assign points_train

#assign points_test

# assign values to dataset class the 
#dataset_text = data_set_model(x_train = points_train, x_test = points_test, options = [])
options_data = {}
options_data['label_path'] = 'D:/Codes-Main-Python/cs699/Project/text_data/radiotalk_labels.csv'
options_data['rep_path'] = 'D:/Codes-Main-Python/cs699/Project/text_data/bert_4.npy'
dataset_to_train = data_set_text(options = options_data)
dataset_to_train.load_dataset()
# create bvae options
options = {}
options['dim_latent'] = 10 # dimension of latent variabels should be
options['activation_type'] = 'relu' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options['learning_rate'] = 1e-3 # what is learning rate (usually 0.001 to 0.01)
options['num_epochs'] = 100 # number of epochs
options['batch_size'] = 500 # number of batches in trainig
options['iteration_plot_show'] = 1000 # when to plot the results
options['print_info_step'] = 1 # how many steps to show info
options['sw_plot'] = 1 # switch for plotting the results within trials
options['plot_mode'] = 'save' #'save' or 'show'
options['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options['train_restore'] = 'restore' # 'train' or 'resotre'
options['dropout_rate'] = 0.3
options['save_global_directory'] = 'D:/Codes-Main-Python/cs699/Project/results_bert'
options['index_latent_supervised'] = np.arange(1) # this specifies the index in latent factors to be correlated with supervised target

capacity_schedule = np.concatenate( ( np.zeros((10))  ,  np.linspace(0,25,options['num_epochs'] - 20) , 25 * np.ones((10)) )  )
sdvaemodel = sdvae_model(data_set = dataset_to_train, gamma = 0, capacity_schedule = capacity_schedule,mu = 50, options = options)

if options['train_restore'] is 'train':
    sdvaemodel.build_computation_graph()
    sdvaemodel.train_model()
elif options['train_restore'] is 'restore':
    sdvaemodel.load_model(epoch_num = 100)
    # load SDVAE train
    z_train, log_sigma2_train = sdvaemodel.project_to_latent(dataset_to_train.x_train)
    z_sup_train = z_train[:,options['index_latent_supervised']]
    y_train = dataset_to_train.y_train
    svm_classifier = SVC(gamma='auto')
    svm_classifier.fit(z_sup_train,y_train)
    #test
    z_test, log_sigma2_test = sdvaemodel.project_to_latent(dataset_to_train.x_test)
    z_sup_test = z_test[:,options['index_latent_supervised']]
    y_test = dataset_to_train.y_test
    y_test_hat = svm_classifier.predict(z_sup_test)
    auc_score = roc_auc_score(y_test,y_test_hat)