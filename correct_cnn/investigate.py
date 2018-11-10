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
#---------------------------------------------------------------------------------------------------------------------------------
'''DEFINE ACTIONS'''
TRAIN = False
TEST = False
WRITE = False
FOM = False
SOMCALC = False
SOMVIS = False

if '--train' in sys.argv:
  TRAIN = True
if '--test' in sys.argv:
  TEST = True
if '--write' in sys.argv:
  WRITE = True
if '--fom' in sys.argv:
  FOM = True
if '--somcalc' in sys.argv:
  SOMCALC = True
if '--somvis' in sys.argv:
  SOMVIS = True

#---------------------------------------------------------------------------------------------------------------------------------
'''DEFINE LOOP'''
trials = ['glna', 'trna', 'riboswitch']

datafiles = {'glna': ['glna_100k_d8.hdf5', '../../data_RFAM/glnAsim_100k.sto'], 
              'trna': ['trna_100k_d4.hdf5', '../../data_RFAM/trnasim_100k.sto'],
              'riboswitch': ['riboswitch_100k_d4.hdf5', '../../data_RFAM/riboswitch_100k.sto'],}

exp = 'corvarfam'  #for both the data folder and the params folder
exp_data = 'data_RFAM'

img_folder = 'Images'

for t in trials:


  #---------------------------------------------------------------------------------------------------------------------------------

  '''OPEN DATA'''

  starttime = time.time()

  #Open data from h5py
  filename = datafiles[t][0]
  data_path = os.path.join('../..', exp_data, filename)
  with h5py.File(data_path, 'r') as dataset:
      X_data = np.array(dataset['X_data'])
      Y_data = np.array(dataset['Y_data'])
      
  numdata, seqlen, _, dims = X_data.shape
  dims = dims-1

  #remove gaps from sequences
  ungapped = True
  if ungapped:
      X_data = X_data[:, :, :, :dims]
      
  # get validation and test set from training set
  test_frac = 0.3
  valid_frac = 0.1
  N = numdata
  split_1 = int(N*(1-valid_frac-test_frac))
  split_2 = int(N*(1-test_frac))
  shuffle = np.random.permutation(N)

  def unalign(X):
    nuc_index = np.where(np.sum(X, axis=2)!=0)
    return (X[nuc_index])

  X_data_unalign = [unalign(X) for X in X_data]
  lengths = [len(X) for X in X_data_unalign]
  maxlength = helper.get_maxlength(X_data_unalign)
  X_data_unalign = np.expand_dims(helper.pad_inputs(X_data_unalign, MAX=maxlength)[0], axis=2)



  #set up dictionaries
  train = {'inputs': X_data_unalign[shuffle[:split_1]], 
           'targets': Y_data[shuffle[:split_1]]}
  valid = {'inputs': X_data_unalign[shuffle[split_1:split_2]], 
           'targets': Y_data[shuffle[split_1:split_2]]}
  test = {'inputs': X_data_unalign[shuffle[split_2:]], 
           'targets': Y_data[shuffle[split_2:]]}

  #set up dictionaries
  train_align = {'inputs': X_data[shuffle[:split_1]], 
           'targets': Y_data[shuffle[:split_1]]}
  valid_align = {'inputs': X_data[shuffle[split_1:split_2]], 
           'targets': Y_data[shuffle[split_1:split_2]]}
  test_align = {'inputs': X_data[shuffle[split_2:]], 
           'targets': Y_data[shuffle[split_2:]]}
      
  print ('Data extraction and dict construction completed in: ' + mf.sectotime(time.time() - starttime))

  print (t, 'Length of Alignment:', X_data.shape[1])
  print (t, 'Max length of all sequences:', X_data_unalign.shape[1])
  print (t, 'Average length of all sequences:', np.mean(lengths))

  simalign_file = datafiles[t][1]
  #Get the full secondary structure and sequence consensus from the emission
  SS = mf.getSSconsensus(simalign_file)
  SQ = mf.getSQconsensus(simalign_file)

  #Get the ungapped sequence and the indices of ungapped nucleotides
  _, ugSS, ugidx = mf.rm_consensus_gaps(X_data, SS)
  _, ugSQ, _ = mf.rm_consensus_gaps(X_data, SQ)


  #Get the sequence and indices of the conserved base pairs
  bpchars = ['(',')','<','>','{','}']
  sig_bpchars = ['<','>']
  bpidx, bpSS, nonbpidx = mf.sigbasepair(SS, bpchars)
  numbp = len(bpidx)
  numug = len(ugidx)

  #Get the bpug information
  bpugSQ, bpugidx = mf.bpug(ugidx, bpidx, SQ)
  #---------------------------------------------------------------------------------------------------------------------------------


  '''SAVE PATHS AND PARAMETERS'''
  params_results = '../../results'

  modelarch = 'resbind'
  trial = t + '_' + exp
  modelsavename = '%s_%s'%(modelarch, trial)



  '''BUILD NEURAL NETWORK'''

  def cnn_model(input_shape, output_shape):

      # create model
      layer1 = {'layer': 'input', #41
              'input_shape': input_shape
              }
      layer2 = {'layer': 'conv1d',
              'num_filters': 96,
              'filter_size': 100,
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


  
