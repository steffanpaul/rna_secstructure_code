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
import bpdev as bd
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

exp = 'grp1intron'  #for the params folder


#---------------------------------------------------------------------------------------------------------------------------------

'''OPEN DATA'''

starttime = time.time()

#Open data from h5py
filename = 'grp1intron_full.hdf5'
with h5py.File(filename, 'r') as dataset:
    X_data = np.array(dataset['X_data'])
    Y_data = np.array(dataset['Y_data'])

numdata, seqlen, _, dims = X_data.shape
dims = dims-1

#remove gaps from sequences
ungapped = True
if ungapped:
    X_data = X_data[:, :, :, :dims]

# get validation and test set from training set
train_frac = 0.8
valid_frac = 0.1
test_frac = 1-0.8-valid_frac
N = numdata
posidx = np.random.permutation(np.arange(N//2))
negidx = np.random.permutation(np.arange(N//2, N))
split_1 = int((N//2)*(1-valid_frac-test_frac))
split_2 = int((N//2)*(1-test_frac))
#shuffle = np.random.permutation(N)

trainidx = np.random.permutation(np.concatenate([posidx[:split_1], negidx[:split_1]]))
valididx = np.random.permutation(np.concatenate([posidx[split_1:split_2], negidx[split_1:split_2]]))
testidx = np.random.permutation(np.concatenate([posidx[split_2:], negidx[split_2:]]))

#set up dictionaries
train = {'inputs': X_data[trainidx],
         'targets': Y_data[trainidx]}
valid = {'inputs': X_data[valididx],
         'targets': Y_data[valididx]}
test = {'inputs': X_data[testidx],
         'targets': Y_data[testidx]}

print ('Data extraction and dict construction completed in: ' + mf.sectotime(time.time() - starttime))

simalign_file = 'grp1intron_full.sto'
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
print (numug, numbp, seqlen)

#---------------------------------------------------------------------------------------------------------------------------------


'''SAVE PATHS AND PARAMETERS'''
params_results = '../../results'

numhidden = int(sys.argv[1])
modelarch = 'mlp_%s'%(str(numhidden))
trial = 'grp1intron_full'
modelsavename = '%s_%s'%(modelarch, trial)

img_folder = 'Images_mlp'#%s'%(modelarch) CHANGE
if not os.path.isdir(img_folder):
    os.mkdir(img_folder)

'''BUILD NEURAL NETWORK'''

def cnn_model(input_shape, output_shape):


  # create model
  layer1 = {'layer': 'input', #41
          'input_shape': input_shape
          }

  layer2 = {'layer': 'dense',        # input, conv1d, dense, conv1d_residual, dense_residual, conv1d_transpose,
                                      # concat, embedding, variational_normal, variational_softmax, + more
            'num_units': numhidden,
            'norm': 'batch',          # if removed, automatically adds bias instead
            'activation': 'relu',     # or leaky_relu, prelu, sigmoid, tanh, etc
            'dropout': 0.5,           # if removed, default is no dropout
           }

  layer3 = {'layer': 'dense',
          'num_units': output_shape[1],
          'activation': 'sigmoid'
          }

  model_layers = [layer1, layer2, layer3]

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
if not os.path.isdir(save_path):
    os.mkdir(save_path)
param_path = os.path.join(save_path, modelsavename)
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=param_path)



#---------------------------------------------------------------------------------------------------------------------------------

'''TRAIN '''
if TRAIN:
  # initialize session
  sess = utils.initialize_session()

  #Train the model

  data = {'train': train, 'valid': valid}
  fit.train_minibatch(sess, nntrainer, data,
                    batch_size=128,
                    num_epochs=2000,
                    patience=1000,
                    verbose=2,
                    shuffle=True,
                    save_all=False)


  sess.close()

  #---------------------------------------------------------------------------------------------------------------------------------
'''TEST'''
sess = utils.initialize_session()
if TEST:

  # set best parameters
  nntrainer.set_best_parameters(sess)

  # test model
  loss, mean_vals, std_vals = nntrainer.test_model(sess, test, name='test')

'''SORT ACTIVATIONS'''
nntrainer.set_best_parameters(sess)
predictionsoutput = nntrainer.get_activations(sess, test, layer='output')
plot_index = np.argsort(predictionsoutput[:,0])[::-1]

#---------------------------------------------------------------------------------------------------------------------------------
'''FIRST ORDER MUTAGENESIS'''
if FOM:
  plots = 3
  num_plots = range(plots)
  fig = plt.figure(figsize=(15,plots*2+1))
  for ii in num_plots:

      X = np.expand_dims(test['inputs'][plot_index[10000+ii]], axis=0)

      ax = fig.add_subplot(plots, 1, ii+1)
      mf.fom_saliency_mul(X, layer='dense_1_bias', alphabet='rna', nntrainer=nntrainer, sess=sess, ax =ax)
      fom_file = modelsavename + 'FoM' + '.png'
  fom_file = os.path.join(img_folder, fom_file)
  plt.savefig(fom_file)

  plt.close()
#---------------------------------------------------------------------------------------------------------------------------------
'''SECOND ORDER MUTAGENESIS'''

'''Som calc'''
if SOMCALC:
  num_summary = np.min([500,len(test['inputs'])//2])
  print (num_summary)

  arrayspath = 'Arrays/%s_%s%s_so%.0fk.npy'%(exp, modelarch, trial, num_summary/1000)
  Xdict = test['inputs'][plot_index[:num_summary]]

  mean_mut2 = mf.som_average_ungapped_split(Xdict, ugidx, arrayspath, nntrainer, sess, split=4, progress='short',
                                             normalize=False, save=True, layer='dense_1_bias')

if SOMVIS:
  #Load the saved data
  num_summary = np.min([500,len(test['inputs'])//2])
  arrayspath = 'Arrays/%s_%s%s_so%.0fk.npy'%(exp, modelarch, trial, num_summary/1000)

  denoise = 'norm'
  vmin = 0.
  if '--apc' in sys.argv:
      denoise = 'APC'
      vmin = 0.
      modelsavename = modelsavename + 'APC'
  C = bd.get_wc(arrayspath, numug, dims, bpugSQ, denoise=denoise)

  if '--apc' not in sys.argv:
      C = C - np.mean(C)
      C = C/np.max(C)

  plt.figure(figsize=(25,20))
  sb.heatmap(C,xticklabels=ugSS, yticklabels=ugSS,vmin=vmin, vmax=None, cmap='viridis', linewidth=0.0)
  plt.title('Base Pair scores: %s %s %s'%(exp, modelarch, trial))

  som_file = modelsavename + 'SoM_bpfilter' + '.png'
  som_file = os.path.join(img_folder, som_file)
  plt.savefig(som_file)
  plt.close()

if WRITE:
    import datetime
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    if not os.path.isfile('test_metrics.csv'):
        fd = open('test_metrics.csv', 'a')
        fd.write('Date,Model Architecture,Hidden Layer Size,Trial,Numpos,Loss,Accuracy,AUC-roc,AUC-pr\n')
        fd.close()
    numpos = len(train['inputs'])//2
    metricsline = '%s,%s,%s,%s,%s,%s,%s,%s,%s'%(date, modelarch, numhidden, trial, numpos,
                 loss, mean_vals[0], mean_vals[1], mean_vals[2])
    fd = open('test_metrics.csv', 'a')
    fd.write(metricsline+'\n')
    fd.close()
