from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from Bio import AlignIO
import sys, h5py
sys.path.append('../../..')
import mutagenesisfunctions as mf
import time as time


modelarch_list = ['glna', 'trna', 'riboswitch']
simalign_file_list = ['../../data_RFAM/glnAsim_100k.sto', '../../data_RFAM/trnasim_100k.sto', '../../data_RFAM/riboswitch_100k.sto']
modeliters = zip(modelarch_list, simalign_file_list)

gappercent_list = [0., 0.25, 0.5, 0.75, 1.0]

for modelarch, simalign_file in modeliters:
    print ('---------------%s-----------------'%(modelarch))

    #Open positive control simulated sequences
    starttime = time.time()
    Xpos = mf.sto_onehot(simalign_file, 'rna')
    Xpos = np.expand_dims(Xpos, axis=2)
    print ('Open positive control: ' + mf.sectotime(time.time()-starttime))

    starttime = time.time()

    ################ PROFILE ##############################
    #Emit sequences from the positive control profile
    Xprofile = np.squeeze(np.mean(Xpos, axis=0))
    numdata, seqlen, _, dims = Xpos.shape
    dims = dims-1
    Xnegprofile = mf.seq_generator_profile(Xprofile, numdata, seqlen, dims)

    print ('Making neg control emitted from frequency profile: '
           + mf.sectotime(time.time() - starttime))

    ################ MAKE NEG WITH GAPCOPIES #######################
    #Make negative controls
    starttime = time.time()
    numdata, seqlen, _, dims = Xpos.shape
    dims = dims-1
    SS = mf.getSSconsensus(simalign_file)
    Xnegrandom = mf.seq_generator_gaps(SS, numdata, seqlen, dims, pgaps=(0.,0.))
    print ('Random sequence generation completed in: ' + mf.sectotime(time.time() - starttime))

    starttime = time.time()

    #insert gaps in the negative control where there were gaps in the positive control
    for s in range(Xpos.shape[0]):
        gapidxcopy = np.where(Xpos[s,:,0,4]==1.)[0]
        Xnegrandom[s, gapidxcopy, :, :] = np.array([0., 0. , 0. ,0., 1.])
        
    print ('Making neg control w/ copy of pos control gaps: '
           + mf.sectotime(time.time() - starttime))

    #check
    if np.sum(Xnegrandom[:,:,:,4]) == np.sum(Xpos[:,:,:,4]):
        print ('Successful gap addition')

    ################ COMBINE ######################
    for gap in gappercent_list:
        print ('=================gappercent: %s================='%(str(gap)))
        #percent of negative controls with gaps
        numdata, seqlen, _, dims = Xpos.shape
        dims = dims-1
        gapportion = np.random.permutation(numdata)[:int(numdata*gap)]
        Xneg = np.concatenate((Xnegprofile[:int(numdata*(1-gapportion))], Xnegrandom[gapportion]))

        #rejoin pos and neg controls
        X_data = np.concatenate((Xpos, Xneg), axis=0)
        numdata, seqlen, _, dims = X_data.shape
        dims = dims-1

        ################ LABELS ####################
        #make Y data
        Y_data = np.zeros((numdata, 1))
        Y_data[:numdata//2, :] = 1.

        starttime = time.time()

        ################ SAVE #######################
        #Save dictionaries into h5py files
        savepath = '../../data_background/%s_100k_gap%0.f.hdf5'%(modelarch, shuff*100)
        with h5py.File(savepath, 'w') as f:
            f.create_dataset('X_data', data=X_data.astype(np.float32), compression='gzip')
            f.create_dataset('Y_data', data=Y_data.astype(np.float32), compression='gzip')
        print ('Saving data: ' + mf.sectotime(time.time() - starttime))
        print ('Saving to: %s'%(savepath))