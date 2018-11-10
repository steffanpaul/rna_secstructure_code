from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import scipy

import sys
sys.path.append('../../..')
import mutagenesisfunctions as mf
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

from Bio import AlignIO
import time as time
import pandas as pd
np.random.seed(274)



def make_variable_length(X_train_fixed, addon=5):
    N = len(X_train_fixed)
    start_buffer = np.random.randint(1, addon, size=N)
    end_buffer = np.random.randint(1, addon, size=N)

    X_train_all = []
    starts = []
    for n in range(N):
        start_size = start_buffer[n]
        start_index = np.random.randint(0,4, size=start_size)
        start_one_hot = np.zeros((start_size, 4))
        for i in range(start_size):
            start_one_hot[i, start_index[i]] = 1
        starts.append(len(start_index))
        
        end_size = end_buffer[n]
        end_index = np.random.randint(0,4, size=end_size)
        end_one_hot = np.zeros((end_size, 4))
        for i in range(end_size):
            end_one_hot[i, end_index[i]] = 1
        X_train_all.append(np.vstack([start_one_hot, X_train_fixed[n], end_one_hot]))
    return X_train_all, starts

#---------------------------------------------------------------------------------------------------------------------------------
'''DEFINE LOOP'''
trials = ['med']#['small', 'med', 'large']
varlengths = [10, 20, 30]
exp = 'toyhp'  #for both the data folder and the params folder
exp_data = 'data_%s'%(exp)

for t in trials:
  for v in varlengths:
    #---------------------------------------------------------------------------------------------------------------------------------

    '''OPEN DATA'''

    starttime = time.time()

    #Open data from h5py
    filename = '%s_50k_%s.hdf5'%(exp, t)
    data_path = os.path.join('../..', exp_data, filename)
    with h5py.File(data_path, 'r') as dataset:
        X_data = np.array(dataset['X_data'])
        Y_data = np.array(dataset['Y_data'])
        
    numdata, seqlen, dims = X_data.shape


    #Make variable!
    addon = v
    X_data, starts = make_variable_length(X_data, addon=addon)
    #pad
    X_data, _ = helper.pad_inputs(X_data, MAX=helper.get_maxlength(X_data))

    X_data = np.expand_dims(X_data, axis=2)
    # get validation and test set from training set
    test_frac = 0.3
    valid_frac = 0.1
    N = numdata
    split_1 = int(N*(1-valid_frac-test_frac))
    split_2 = int(N*(1-test_frac))
    shuffle = np.random.permutation(N)

    #set up dictionaries
    train = {'inputs': X_data[shuffle[:split_1]], 
             'targets': Y_data[shuffle[:split_1]]}
    valid = {'inputs': X_data[shuffle[split_1:split_2]], 
             'targets': Y_data[shuffle[split_1:split_2]]}
    test = {'inputs': X_data[shuffle[split_2:]], 
             'targets': Y_data[shuffle[split_2:]]}
    test_starts = np.asarray(starts)[shuffle[split_2:]]
    print ('Data extraction and dict construction completed in: ' + mf.sectotime(time.time() - starttime))


    #---------------------------------------------------------------------------------------------------------------------------------


    '''SAVE PATHS AND PARAMETERS'''
    params_results = '../../results'

    modelarch = 'resbind'
    trial = 'var' + str(v) + t
    modelsavename = '%s_%s'%(modelarch, trial)



    '''BUILD NEURAL NETWORK'''

    def cnn_model(input_shape, output_shape):

        # create model
        layer1 = {'layer': 'input', #41
                'input_shape': input_shape
                }
        layer2 = {'layer': 'conv1d',
                'num_filters': 96,
                'filter_size': input_shape[1]-29,
                'norm': 'batch',
                'activation': 'relu',
                'dropout': 0.3,
                'padding': 'VALID',
                }
        layer3 = {'layer': 'conv1d_residual',
                'filter_size': 5,
                'function': 'relu',
                'dropout_block': 0.1,
                'dropout': 0.3,
                'mean_pool': 10,
                }
        
        layer4 = {'layer': 'dense',        # input, conv1d, dense, conv1d_residual, dense_residual, conv1d_transpose,
                                        # concat, embedding, variational_normal, variational_softmax, + more
              'num_units': 196,
              'norm': 'batch',          # if removed, automatically adds bias instead
              'activation': 'relu',     # or leaky_relu, prelu, sigmoid, tanh, etc
              'dropout': 0.5,           # if removed, default is no dropout
                 }

        
        layer5 = {'layer': 'dense',
                'num_units': output_shape[1],
                'activation': 'sigmoid'
                }

        model_layers = [layer1, layer2, layer3, layer4, layer5]

        # optimization parameters
        optimization = {"objective": "binary",
                      "optimizer": "adam",
                      "learning_rate": 0.0003,
                      "l2": 1e-5,
                      #"label_smoothing": 0.05,
                      #"l1": 1e-6,
                      }
        return model_layers, optimization

    tf.reset_default_graph()

    # get shapes of inputs and targets
    input_shape = list(train['inputs'].shape)
    input_shape[0] = None
    output_shape = train['targets'].shape

    # load model parameters
    model_layers, optimization = cnn_model(input_shape, output_shape)

    # build neural network class
    nnmodel = nn.NeuralNet(seed=247)
    nnmodel.build_layers(model_layers, optimization)

    # compile neural trainer
    save_path = os.path.join(params_results, exp)
    param_path = os.path.join(save_path, modelsavename)
    nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=param_path)

    #---------------------------------------------------------------------------------------------------------------------------------

    '''TRAIN '''

    # initialize session
    sess = utils.initialize_session()

    #Train the model

    data = {'train': train, 'valid': valid}
    fit.train_minibatch(sess, nntrainer, data, 
                        batch_size=100, 
                        num_epochs=100,
                        patience=40, 
                        verbose=2, 
                        shuffle=True, 
                        save_all=False)


    sess.close()

    #---------------------------------------------------------------------------------------------------------------------------------


