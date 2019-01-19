from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from Bio import AlignIO
import sys, h5py, os
sys.path.append('../../..')
import mutagenesisfunctions as mf
import time as time

'''Steps involved in making hdf5 data file:
    - convert .sto to one hot file
    - Add pseudo dimension
    - Make PWM profile distribution
    - Emit an equal number of sequences from distribution
    - Concatenate the files
    - Generate Y labels
    - Generate metadata dictionary!
    - Save data
'''

#retrieve stockholm files
famnames = next(os.walk('.'))[1] #incantation to get subdirectories of current directory - also corresponds to names of data

#loop over families
for fam in famnames:
    simalign_file = '%s/%s.sto'%(fam, fam) #Zasha Weinberg .sto files are just called their name

    print ('---------------%s-----------------'%(fam))

    #Open positive control simulated sequences
    starttime = time.time()
    Xpos = mf.sto_onehot(simalign_file, 'rna')
    Xpos = np.expand_dims(Xpos, axis=2)

    #secondary structure information
    SS = mf.getSSconsensus(simalign_file)

    #sequence statistics of unaligned data
    Xpos_unalign = mf.unalign_full(Xpos[:, :, :, :-1])
    lengths = [len(x) for x in Xpos_unalign]
    mean_len = np.mean(lengths)
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    std_len = np.std(lengths)

    print ('Open positive control: ' + mf.sectotime(time.time()-starttime))

    starttime = time.time()

    ################ PROFILE ##############################
    #Emit sequences from the positive control profile
    Xprofile = np.squeeze(np.mean(Xpos, axis=0))
    numpos, seqlen, _, dims = Xpos.shape
    print ('numpos:', numpos, 'seqlen:', seqlen)
    dims = dims-1
    Xnegprofile = mf.seq_generator_profile(Xprofile, numpos, seqlen, dims)

    pet = mf.sectotime(time.time() - starttime) #profile emission time
    print ('Making neg control emitted from frequency profile: '
       + pet)

    starttime = time.time()

    #rejoin pos and neg controls
    X_data = np.concatenate((Xpos, Xnegprofile), axis=0)
    numdata, seqlen, _, dims = X_data.shape

    ################ LABELS ####################
    #make Y data
    Y_data = np.zeros((numdata, 1))
    Y_data[:numdata//2, :] = 1.

    starttime = time.time()

    ################ METADATA ####################
    meta = [['Family Name', fam],
            ['Source', 'Z. Weinberg'],
            ['Alignment depth', numpos],
            ['Alignment length', seqlen],
            ['Mean seq length', mean_len],
            ['Min seq length', min_len],
            ['Max seq length', max_len],
            ['Std seq length', std_len],
            ['Profile emission time', pet]]

    #print the metadata in a nice way
    fd = open('zwein_data_details.txt', 'a')
    for item in meta:
        fd.write('{0:25} {1:}\n'.format(item[0], item[1]))
    fd.write('\n')
    fd.write(SS)
    fd.write('\n \n')
    fd.close()

    ################ SAVE #######################
    #Save dictionaries into h5py files
    savepath = '%s/%s_full.hdf5'%(fam, fam)
    with h5py.File(savepath, 'w') as f:
        f.create_dataset('X_data', data=X_data.astype(np.float32), compression='gzip')
        f.create_dataset('Y_data', data=Y_data.astype(np.float32), compression='gzip')
        f.create_dataset('metadata', data=meta, compression='gzip')
    print ('Saving data: ' + mf.sectotime(time.time() - starttime))
    print ('Saving to: %s'%(savepath))
