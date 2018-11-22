from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np

from shutil import copyfile
sys.path.append('../../../..')
import bpdev as bd


def import_pretransfer(params_results, exp, datatype, modelarch, truemodelsavename, isrnn=False):

    '''
    Define a function that makes a copy of the pretansfer names
    as the current trial path filenames. This will allow new models to
    be trained starting from the same pretranfer paramaters.

    Inputs: params_results (the path of the results folder)
            exp (the experiment we're currently on)
            datatype (the dataselection we're currently using)
            trialnum (the trial number we're on - for the newfile
                        generation)
            modelarch (the modelarchitecture we're using)

    Outputs: doesn't return anything, just performs the copy so the model
                can start training.

    '''


    #Make the pretransfer file a copy renamed as the current trial path

    #First make a list of the pretransfer filenames
    trial = 'pkhp_d%s_pretran'%(datatype)
    modelsavename = '%s_%s'%(modelarch, trial)

    save_path = os.path.join(params_results, exp)
    param_path = os.path.join(save_path, modelsavename)

    oldfiles = ['%s_best.ckpt.data-00000-of-00001'%(param_path), '%s_best.ckpt.index'%(param_path), '%s_best.ckpt.meta'%(param_path)]
    if isrnn:
        oldfiles = ['%s_best.data-00000-of-00001'%(param_path), '%s_best.index'%(param_path), '%s_best.meta'%(param_path)]

    #Now make a list of the trial filenames
    #trial = 'pkhp_d%st%s'%(datatype, trialnum)
    #modelsavename = '%s_%s'%(modelarch, trial)

    save_path = os.path.join(params_results, exp)
    param_path = os.path.join(save_path, truemodelsavename)

    newfiles = ['%s_best.ckpt.data-00000-of-00001'%(param_path), '%s_best.ckpt.index'%(param_path), '%s_best.ckpt.meta'%(param_path)]
    if isrnn:
        newfiles = ['%s_best.data-00000-of-00001'%(param_path), '%s_best.index'%(param_path), '%s_best.meta'%(param_path)]
    #Now make the copy
    for ii in range(len(newfiles)):
        copyfile(oldfiles[ii], newfiles[ii])

#function that outputs the necessary paramaters for the toypkhp
def pkhp_SS(structure = 'simple'):
    if structure == 'simple':
        SShp = '.'*5 +'('*5 + '.'*7 +  ')'*5 + '.'*5 + '.'*5 +'('*5 + '.'*7 +  ')'*5 + '.'*5
        SSpk = '.'*12 +'('*3 + '.'*24 +  ')'*3 + '.'*12
    ugSS = [SShp, SSpk]
    numbp = len(bd.bp_coords(ugSS))
    numug = len(ugSS[0])
    bpugSQ = [str(i) for i in range(numug)]

    return (ugSS, numbp, numug, bpugSQ)
