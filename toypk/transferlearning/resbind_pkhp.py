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
import helptransfer as htf
sys.path.append('../../../..')
import mutagenesisfunctions as mf
import bpdev as bd
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

#from Bio import AlignIO
import time as time
import pandas as pd
#---------------------------------------------------------------------------------------------------------------------------------
'''DEFINE ACTIONS'''
TRAIN = False
TEST = False
WRITE = True
FOM = False
SOMCALC = False
SOMVIS = False
TRANSFER = False
SOME = False
JUSTPKHP = False
PRETRANSFER = False


if '--pretransfer' in sys.argv:
  PRETRANSFER = True
if '--transfer' in sys.argv:
  TRANSFER = True
if '--some' in sys.argv:
  SOME = True
if '--justpkhp' in sys.argv:
  JUSTPKHP = True
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

exp = 'toypk'  #for the params folder
modelarch = 'resbind'

datatype = sys.argv[1]
trialnum = sys.argv[2]
if SOME:
  portion = int(sys.argv[sys.argv.index('--some')+1]) #pulls out the divisor of the data portion eg. 50 = 50,000/50 = 1000 seqs

if '--setepochs' in sys.argv: #set the number of epochs over which the model will train (with no patience)
  numepochs = int(sys.argv[sys.argv.index('--setepochs')+1])
else:
  numepochs = 100

img_folder = 'Images_%s_d%s'%(modelarch, datatype)
if not os.path.isdir(img_folder):
  os.mkdir(img_folder)
#---------------------------------------------------------------------------------------------------------------------------------

'''OPEN DATA'''

starttime = time.time()

#Open data from h5py
exp_data = 'data_toypk'
filename = 'toypkhp_50_d%s.hdf5'%(datatype)
data_path = os.path.join('../../..', exp_data, filename)

if TRANSFER: #import pkhp data to transfer learn
    ext = '_pkhp'
elif JUSTPKHP:
    ext = '_pkhp'
elif PRETRANSFER:
    ext = '_hp'

with h5py.File(data_path, 'r') as dataset:
    X_pos = np.array(dataset['X_pos%s'%(ext)])
    X_neg = np.array(dataset['X_neg%s'%(ext)])

    Y_pos = np.array(dataset['Y_pos'])
    Y_neg = np.array(dataset['Y_neg'])


X_pos = np.expand_dims(X_pos, axis=2)
X_neg = np.expand_dims(X_neg, axis=2)

numdata, seqlen, _, dims = X_pos.shape


if not SOME:
    X_data = np.concatenate((X_pos, X_neg), axis=0)
    Y_data = np.concatenate((Y_pos, Y_neg), axis=0)
if SOME:

    X_data = np.concatenate((X_pos[:numdata//portion], X_neg[:numdata//portion]), axis=0)
    Y_data = np.concatenate((Y_pos[:numdata//portion], Y_neg[:numdata//portion]), axis=0)
# get validation and test set from training set
if not TRANSFER: #set the proportions for pretransfer
    train_frac = 0.5 #This means the pretransfer model is training on 25,000 pos and 25,000 neg sequences
    valid_frac = 0.2
    test_frac = 0.3
if TRANSFER or JUSTPKHP:
    train_frac = 0.8
    valid_frac = 0.1
    test_frac = 0.1

numdata, seqlen, _, dims = X_data.shape

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
#---------------------------------------------------------------------------------------------------------------------------------


'''SAVE PATHS AND PARAMETERS'''
params_results = '../../../results'

trial = 'pkhp_d%st%s'%(datatype, trialnum)
if '--setepochs' in sys.argv:
  trial = 'pkhp_d%st%se%s'%(datatype, trialnum, numepochs)

if PRETRANSFER:
  trial = 'pkhp_d%s_pretran'%(datatype)
  numepochs = 100


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
nnmodel = nn.NeuralNet(seed=42)
nnmodel.build_layers(model_layers, optimization)

# compile neural trainer
save_path = os.path.join(params_results, exp)
param_path = os.path.join(save_path, modelsavename)
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=param_path)

#SET UP A CLAUSE TO INITIATE TRANSFER LEARNING
if TRANSFER and TRAIN:
    #make the pretransfer file a copy of what we want now
    htf.import_pretransfer(params_results, exp, datatype, modelarch, modelsavename)



#---------------------------------------------------------------------------------------------------------------------------------

'''TRAIN '''
if TRAIN:
    # initialize session
    sess = utils.initialize_session()

    if TRANSFER:
      # set best parameters to transfer learning
        nntrainer.set_best_parameters(sess)

    #Train the model

    data = {'train': train, 'valid': valid}
    fit.train_minibatch(sess, nntrainer, data,
                    batch_size=100,
                    num_epochs=numepochs,
                    patience=numepochs,
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
    X = np.expand_dims(test['inputs'][plot_index[ii]], axis=0)

    ax = fig.add_subplot(plots, 1, ii+1)
    mf.fom_saliency_mul(X, layer='dense_1_bias', alphabet='rna', nntrainer=nntrainer, sess=sess, ax =ax)

    if not TRANSFER:
        fom_file = modelsavename + 'FoM' + '_pretransfer' + '.png'
    else:
        fom_file = modelsavename + 'FoM' + '_posttransfer' + '.png'
  fom_file = os.path.join(img_folder, fom_file)
  plt.savefig(fom_file)

  plt.close()
#---------------------------------------------------------------------------------------------------------------------------------
'''SECOND ORDER MUTAGENESIS'''

'''Som calc'''
if SOMCALC:
  num_summary = np.min([500,len(test['inputs'])//2])

  arrayspath = 'Arrays_cnn/%s_%s_so%.0fk.npy'%(exp, modelsavename, num_summary/1000)
  Xdict = test['inputs'][plot_index[:num_summary]]

  mean_mut2 = mf.som_average_ungapped_logodds(Xdict, range(seqlen), arrayspath, nntrainer, sess, progress='short',
                                             save=True, layer='dense_1_bias')

if SOMVIS:
  #Load the saved data
  num_summary = np.min([500,len(test['inputs'])//2])
  arrayspath = 'Arrays_cnn/%s_%s_so%.0fk.npy'%(exp, modelsavename, num_summary/1000)
  mean_mut2 = np.load(arrayspath)

  #Reshape into a holistic tensor organizing the mutations into 4*4
  meanhol_mut2 = mean_mut2.reshape(seqlen,seqlen,4,4)

  #Normalize
  normalize = True
  if normalize:
      norm_meanhol_mut2 = mf.normalize_mut_hol(meanhol_mut2, nntrainer, sess, normfactor=1)

  #Let's try something weird
  bpfilter = np.ones((4,4))*0.
  for i,j in zip(range(4), range(4)):
      bpfilter[i, -(j+1)] = +1.

  nofilter = np.ones((4,4))

  C = (norm_meanhol_mut2*bpfilter)
  C = np.sum((C).reshape(seqlen,seqlen,dims*dims), axis=2)
  #C = C - np.mean(C)
  #C = C/np.max(C)
  ugSS, numbp, numug, bpugSQ = htf.pkhp_SS()
  if datatype == '6' and not TRANSFER:
      ugSS = ugSS[1] #only extract the non-nested base pairs
      numbp = 3
  #get base pairing scores
  totscore = bd.bp_totscore(ugSS, C, numug)
  ppv = bd.bp_ppv(C, ugSS, numbp, numug)

  plt.figure(figsize=(8,6))
  sb.heatmap(C,vmin=None, cmap='RdPu', linewidth=0.0)
  plt.title('Base Pair scores: %s %s '%(exp, modelsavename))

  if JUSTPKHP:
    som_file = modelsavename + 'SoM_bpfilter' + '_justpkhp' + '.png'
  if TRANSFER:
    som_file = modelsavename + 'SoM_bpfilter' + '_posttransfer' + '.png'
  else:
    som_file = modelsavename + 'SoM_bpfilter' + '_pretransfer' +'.png'
  som_file = os.path.join(img_folder, som_file)
  plt.savefig(som_file)
  plt.close()

  if WRITE:
      tran = 'notransfer'
      if TRANSFER:
          tran = 'transfer'
      numpos = len(train['inputs'])//2
      metricsline = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(modelarch, datatype, trialnum, numepochs, tran,
                        numpos, loss, mean_vals[0], mean_vals[1], mean_vals[2], totscore, ppv)
      fd = open('test_metrics.csv', 'a')
      fd.write(metricsline+'\n')
      fd.close()
