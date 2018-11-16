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
sys.path.append('../../../..')
import mutagenesisfunctions as mf
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
WRITE = False
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
modelarch = 'rnn'

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

#I was too lazy to rewrite this code so - pull arrays out of lists and remove pseudo dimension
X_train = train['inputs'][:, :, 0, :]
X_valid = valid['inputs'][:, :, 0, :]
X_test = test['inputs'][:, :, 0, :]

Y_train = train['inputs']
Y_valid = valid['inputs']
Y_test = test['inputs']

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


print (X_train.shape)