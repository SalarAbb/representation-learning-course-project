# this supervised vae class
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
from scipy import stats, io
import tensorflow as tf
from tensorflow.contrib import rnn
from numpy import linalg as LA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# methods for RNN

class supervised_vae_model(object):
    # this class creates the RNN model for movement generation
    
    def __init__(self,data_set,beta,mu,options):
        # options: options of loading the RNN model
        # what are beta and mu?
        # maxmize -> loss = - reconstruction_loss - beta * (kl_divergence) - mu * (supervision_loss - adversarial_loss)
        # we implement this: minimize -> loss = + reconstruction_loss + beta * (kl_divergence) + mu * (supervision_loss - adversarial_loss)
        # we need mu to balance losses values 
        self.options = options
        self.dim_latent = options['dim_latent'] # dimension of latent variabels
        self.activation_type = options['activation_type'] # activation function in the layers 'tanh' or None or 'relu' or 'elu'
        self.learning_rate = options['learning_rate'] # what is learning rate (usually 0.001 to 0.01)
        #self.num_iterations_per_epoch = options['num_iterations_per_epoch'] # number of iterations per epoch
        self.num_epochs = options['num_epochs'] # number of epochs
        self.batch_size = options['batch_size'] # number of batches in trainig
        self.print_info_step = options['print_info_step'] # how many steps to show info
        self.sw_plot = options['sw_plot'] # switch for plotting the results within trials
        self.plot_mode = options['plot_mode'] # 'show' to show the plot or 'save' to only save the plot
        self.optimizer_type = options['optimizer_type'] # 'Adam' or 'GD'
        self.train_restore = options['train_restore'] # whether we learn or restore (maybe delete later)
        self.iteration_plot_show = options['iteration_plot_show']
        self.dropout_rate = options['dropout_rate']
        self.save_global_directory = options['save_global_directory']
        self.index_latent_supervised = options['index_latent_supervised'] # dimension of latent variable that is related to supervised variables
        self.index_latent_adversarial = np.arange(self.dim_latent)
        self.index_latent_adversarial = self.index_latent_adversarial.delete(self.index_latent_supervised)
        # supervison variables
        self.y_train = data_set.y_train
        self.y_test = data_set.y_test
        self.dim_y = data_set.dim_y
        # create train and test dataset
        self.dim_x = data_set.dim_x
        self.x_train = data_set.x_train
        self.x_train_num_samples = np.shape(self.x_train)[0]

        self.x_test = data_set.x_test
        self.x_test_num_samples = np.shape(self.x_test)[0]

        
        #set additional parameters
        self.num_iterations_per_epoch = np.int(np.floor(self.x_train_num_samples/self.batch_size))
        # beta is a scalar or is beta schedule - > in each case create beta schedule
        self.beta = beta
        self.mu = mu
        if np.isscalar(self.beta):
            self.flag_beta_scalar = 1
            self.beta_schedule = self.beta * np.ones((self.num_epochs) , dtype='float32')
        else:
            self.flag_beta_scalar = 0
            self.beta_schedule = self.beta
        # self.save_directory = options['save_directory'] # directory to save the results
        self.create_directory()

    def create_directory(self):
        # self.save_folder_name = options['save_folder_name'] # file name to save the results  
        string_info = ''

        string_info = string_info + '_dimlat{}'.format(self.dim_latent)


        self.save_main_folder = '{}/beta_vae/trained_model_beta{:.2f}_mu{:.2f}{}'.format(self.save_global_directory, self.beta, self.mu, string_info)


        return    

    def recognition_network(self,recog_input):
        # we want to build this recognition network:
        # these are the hidden layers -> input-10-10-10-mu,sigma
        
        initilizer_dense = tf.contrib.layers.xavier_initializer(uniform=False)


        net = recog_input

        net = tf.layers.dense(net,20, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net,20, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net,20, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        mu = tf.layers.dense(net,self.dim_latent, kernel_initializer = initilizer_dense)
        # we need to design mu more intelligently, mu sometimes does get loop coordinates, therefore we should take that into account
        # therefore we force all latent variabales to be around 
        mu_0 = tf.expand_dims( tf.acos(mu[:,0]) , 1)
        mu_1 = tf.expand_dims( mu[:,1] , 1)

        mu = tf.concat( [mu_0,mu_1] , axis = 1 )

        # sigma2  = tf.layers.dense(net,self.dim_latent, kernel_initializer = initilizer_dense)
        # sigma2 = tf.math.square(sigma2)
        # log_sigma2 = tf.math.log(sigma2+1e-8)
        log_sigma2 = tf.layers.dense(net,self.dim_latent, kernel_initializer = initilizer_dense)
        

        return mu, log_sigma2

    def generative_network(self,latent_sample):
        # we want to build this recognition network:
        # sample-10-10-10-x_recons
        
        initilizer_dense = tf.contrib.layers.xavier_initializer(uniform=False)
        
        net = latent_sample


        net = tf.layers.dense(net,20, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)
    
        net = tf.layers.dense(net,20, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net,20, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net,self.dim_x, kernel_initializer = initilizer_dense)
        x_reconstruct = net

        return x_reconstruct

    def classification_network(self,latent_sample,list_hidden_units):
        # we want to build this recognition network:
        # list_hidden_units
        initilizer_dense = tf.contrib.layers.xavier_initializer(uniform=False)
        if self.activation_type == 'relu':
            non_linear = lambda x: tf.nn.relu(x)
        elif self.activation_type == 'tanh':
            non_linear = lambda x: tf.nn.tanh(x)
        
        for hidden in list_hidden_units:
            net = tf.layers.dense(net,hidden, kernel_initializer = initilizer_dense)
            net = non_linear(net)

        net = tf.layers.dense(net,self.dim_y, kernel_initializer = initilizer_dense)

        return net
    

    def build_computation_graph(self):
        self.sess = tf.Session()
        # define auto encoder architecture
        self.x = tf.placeholder(tf.float32 , shape=[None,self.dim_x] ) # num_batches, smaples in each batch
        self.y = tf.placeholder(tf.float32 , shape=[None,self.dim_y] ) # num_batches, smaples in each batch 
        self.label = tf.one_hot(self.y ,depth = self.dim_y,dtype='int32') 
        self.lr = tf.placeholder(tf.float32 , shape=[] )
        self.is_training = tf.placeholder(tf.bool, [])
        self.beta = tf.placeholder(tf.float32, [])
        self.latent_mu, self.latent_log_sigma2 = self.recognition_network(self.x)

        eps = tf.random_normal(tf.shape(self.latent_log_sigma2), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
                       
        self.latent_sample = self.latent_mu + tf.exp(self.latent_log_sigma2/2) * eps
        # from latent sample we build three networks
        # 1. generative networks
        # 2. calssification_supervised_index_network networks
        # 3. calssification_adverarial_index_network networks
        
        # let's use the latent sample to reconstruct
        self.y_hat_supervised = self.classification_network(self.latent_sample[:,self.index_latent_supervised], list_hidden_units = [40,40,40])
        self.y_hat_adversarial = self.classification_network(self.latent_sample[:,self.index_latent_adversarial], list_hidden_units = [100,100,100]) 
        # let's reconstric
        self.x_recons = self.generative_network(self.latent_sample)


        # define loss function
        self.loss_train = self.compute_loss()
        

        # define optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)    
        elif self.optimizer_type == 'GD':
            self.optimizer_train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)   

        # define operation of training
        self.operation_train = self.optimizer_train.minimize(self.loss_train)

        # initlize the training variables values
        self.init = tf.global_variables_initializer() 
        # saver for later work
        self.saver = tf.train.Saver()

        return       
    def gauss_sample(self,mu, var):
        epsilon = tf.random_normal(tf.shape(var), name="epsilon")
        return mu + tf.sqrt(var) * epsilon

    def compute_loss(self):
        # Reconstruction loss
        recons_loss = tf.math.square(self.x - self.x_recons)
        recons_loss = tf.reduce_sum(recons_loss,axis=1)
        self.recons_loss = tf.reduce_mean(recons_loss)
        # Latent loss
        latent_loss = -0.5 * tf.reduce_sum(1 + self.latent_log_sigma2
                                        - tf.square(self.latent_mu)
                                        - tf.exp(self.latent_log_sigma2), 1)
        self.latent_loss = tf.reduce_mean(latent_loss)   
        #  supervised and adversarial classification loss
        self.loss_supervised = self.compute_loss_classification(self.y_hat_supervised,self.label)
        self.loss_adversarial = self.compute_loss_classification(self.y_hat_adversarial,self.label)   
        # Loss with encoding capacity term
        return self.recons_loss + self.beta * self.latent_loss + self.mu * (self.loss_supervised - self.loss_adversarial)
    
    def compute_loss_classification(self,network_output,label):
       # label is class labels for samples: [batch_size,label]
       # network_output is the output of the network: 
       # NOTE: network output should not pass through softmax as tf.nn.softmax_cross_entropy_with_logits_v2 does it.

        labels_float32 = tf.cast(label,tf.float32)
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_float32,logits=network_output)
        loss = tf.reduce_mean( cross_entropy_loss)
        return loss
    
    def log_gaussian(self,x, mu, log_sigma2):
        const_log_pdf = (- 0.5 * np.log(2 * np.pi)).astype('float32') 
        #return const_log_pdf - tf.log(var) / 2 - tf.square(x - mu) / (2 * var)
        return const_log_pdf - log_sigma2 / 2 - tf.square((x - mu)) / (2 * tf.exp(log_sigma2))
     
    def train_model (self):
        # iterations to plot

        # create the loss_list
        self.loss_tot_train = []
        self.loss_recons_train = []
        self.loss_latent_train = []
        self.loss_tot_test = []
        self.loss_recons_test = []
        self.loss_latent_test = []
        #self.loss_val_iter = []


        if self.train_restore == 'train':        
            # initilize
            self.sess.run(self.init)
            for epoch in range(1,self.num_epochs+1):
                # create the correct beta
                beta_this = self.beta_schedule[epoch-1] # assign the correct beta to the schedule
                #
                shuffle_index = np.random.randint(self.x_train_num_samples, size=(self.x_train_num_samples))
        
                for iteration in range(1,self.num_iterations_per_epoch+1):
                    batch_index = shuffle_index[ 
                        range( (iteration-1)*self.batch_size , (iteration)*self.batch_size) ]
                    # create the batches of input and target TODO: make it batch based
                    batch_x = self.x_train[batch_index,:]
                    # run gradient descent
                    self.sess.run(self.operation_train, feed_dict = { self.x: batch_x, self.is_training: True, self.beta: beta_this} ) 
                    # plot
                    if self.sw_plot == 1:
                        plot_folder = '{}/plots'.format(self.save_main_folder)
                        plot_name = 'epoch_{}_train.png'.format(epoch)
                        #
                        print('add plot function')
                        #
                    # display training results
                    if iteration % self.print_info_step == 0 or iteration == self.num_iterations_per_epoch:
                        loss_this, recons_loss_this, latent_loss_this= self.sess.run([self.loss_train,self.recons_loss,self.latent_loss], feed_dict = {self.x: batch_x, self.is_training: True, self.beta: beta_this})
                        # print info                        
                        print('TRAIN: epoch {}, iteration {} --> loss_tot = {}, loss_rec = {}, loss_lat = {}'.format(epoch,iteration,loss_this, recons_loss_this, latent_loss_this))    
                        # append loss
                        self.loss_tot_train.append(loss_this)
                        self.loss_recons_train.append(recons_loss_this)
                        self.loss_latent_train.append(latent_loss_this)
                
                # create the batches of input and target TODO: make it batch based
                batch_x_test = self.x_test[:,:]
                # run gradient descent
                loss_this, recons_loss_this, latent_loss_this = self.sess.run([self.loss_train,self.recons_loss,self.latent_loss], feed_dict = {self.x: batch_x_test, self.is_training: False, self.beta: beta_this})
                print('TEST: epoch {}--> loss_tot = {}, loss_rec = {}, loss_lat = {}'.format(epoch,loss_this, recons_loss_this, latent_loss_this))
                self.loss_tot_test.append(loss_this)
                self.loss_recons_test.append(recons_loss_this)
                self.loss_latent_test.append(latent_loss_this)
                #print( 'loss_test so far: {}'.format(self.loss_tot_test) )
                if self.sw_plot == 1:
                        plot_folder = '{}/plots'.format(self.save_main_folder)
                        plot_name = 'epoch_{}_test.png'.format(epoch)
                        ##
                        print('add plot function')
                        ##
                        
                # save the model
                self.save_model(epoch)
                
        return 

    def save_model(self,epoch_num):
        
        save_folder = "{}/epoch_{}".format(self.save_main_folder, epoch_num)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('This directory was created: {}'.format(save_folder))

        file_path = "{}/model.ckpt".format(save_folder)
        print('model is being saved as {}'.format(file_path))
        save_path = self.saver.save(self.sess, file_path)

        return

    def load_model(self, epoch_num):
        model_path = "{}/epoch_{}/model.ckpt".format(self.save_main_folder, epoch_num)
        self.build_computation_graph()
        self.saver.restore(self.sess,model_path)

    def reconstruct(self,x):
        x_recons = self.sess.run(self.x_recons,feed_dict = {self.x: x, self.is_training: False})
        return x_recons

    def project_to_latent(self,x):
        z_mean, z_log_sigma2 = self.sess.run([self.latent_mu, self.latent_log_sigma2],feed_dict = {self.x: x, self.is_training: False})
        return np.asarray(z_mean,dtype='float32'), np.asarray(z_log_sigma2,dtype='float32')

    def reconstruct_from_latent(self,z):
        x_recons = self.sess.run(self.x_recons,feed_dict = {self.latent_sample: z, self.is_training: False})
        return x_recons    

