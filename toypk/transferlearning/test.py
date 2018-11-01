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

from Bio import AlignIO
import time as time
import pandas as pd

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




exp = 'toypk_again'  #for the params folder
modelarch = 'mlp'

datatype = sys.argv[1]
trialnum = sys.argv[2]


'''SAVE PATHS AND PARAMETERS'''
params_results = '../../../results'

trial = 'pkhp_d%st%s'%(datatype, trialnum)

if PRETRANSFER:
  trial = 'pkhp_d%s_pretran'%(datatype)

modelsavename = '%s_%s'%(modelarch, trial)

# compile neural trainer
save_path = os.path.join(params_results, exp)
param_path = os.path.join(save_path, modelsavename)
#nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=param_path)

#SET UP A CLAUSE TO INITIATE TRANSFER LEARNING
if TRANSFER:
    #make the pretransfer file a copy of what we want now
    import helptransfer as htf
    htf.import_pretransfer(params_results, exp, datatype, trialnum, modelarch)
