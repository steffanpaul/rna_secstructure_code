from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from Bio import AlignIO
import sys, h5py
sys.path.append('../../..')
import mutagenesisfunctions as mf
import time as time



def seq_bunchshuffle(Xpos, numdata, seqlen, bunchsize=(10, 75)):

    #n = the number of bunches
    smallbunch, largebunch = bunchsize
    n_upper = seqlen//smallbunch
    n_lower = seqlen//largebunch

    Xshuffle = np.zeros((np.shape(Xpos)))
    ns = []
    for seq in range(numdata):
        Xcopy = np.copy(Xpos[seq])

        n = np.random.randint(n_lower, n_upper)

        bunchidx = [i*(seqlen//n) for i in range(n)]
        bunchidx.append(seqlen)

        start=0
        randidx = np.random.permutation(n)
        for i in range(n):
            idx = randidx[i]
            space = bunchidx[idx+1]-bunchidx[idx]
            Xshuffle[seq, start:start+space, :, :] = Xcopy[bunchidx[idx]:bunchidx[idx+1], :, :]
            start = start + space
            
    return (Xshuffle)


modelarch_list = ['glna', 'trna', 'riboswitch']
simalign_file_list = ['../../data_RFAM/glnAsim_100k.sto', '../../data_RFAM/trnasim_100k.sto', '../../data_RFAM/riboswitch_100k.sto']
modeliters = zip(modelarch_list, simalign_file_list)

shufflepercent_list = [0., 0.25, 0.5, 0.75, 1.0]

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

    starttime = time.time()

    ################ SHUFFLE #######################
    #Create a negative control of randomly suffled sequences
    numdata, seqlen, _, dims = Xpos.shape
    dims = dims-1

    Xnegshuffle = seq_bunchshuffle(Xpos, numdata, seqlen)

    print ('Making neg control as shuffled pos: ' + mf.sectotime(time.time() - starttime))

    ################ COMBINE ######################
    for shuff in shufflepercent_list:
        print ('=================shufflepercent: %s================='%(str(shuff)))
        #percent of shuffled negative controls
        shufflepercent = shuff
        numdata, seqlen, _, dims = Xpos.shape
        dims = dims-1
        shuffleportion = np.random.permutation(numdata)[:int(numdata*shufflepercent)]
        Xneg = np.concatenate((Xnegprofile[:int(numdata*(1-shufflepercent))], Xnegshuffle[shuffleportion]))

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
        savepath = '../../data_background/%s_100k_sh%0.f.hdf5'%(modelarch, shuff*100)
        with h5py.File(savepath, 'w') as f:
            f.create_dataset('X_data', data=X_data.astype(np.float32), compression='gzip')
            f.create_dataset('Y_data', data=Y_data.astype(np.float32), compression='gzip')
        print ('Saving data: ' + mf.sectotime(time.time() - starttime))
        print ('Saving to: %s'%(savepath))