from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from Bio import AlignIO
import os, sys, h5py
sys.path.append('../../..')
import mutagenesisfunctions as mf
import time as time

#retrieve stockholm files
famnames = next(os.walk('.'))[1] #incantation to get subdirectories of current directory - also corresponds to names of data

from collections import Counter
nucs_count = Counter() #counter that will go over every file

for fam in famnames:
    alignment = AlignIO.read(open('%s/%s.sto'%(fam, fam)), "stockholm")
    sequences = [record.seq for record in alignment]
    for seq in sequences:
        for n in seq:
            nucs_count[n]+=1
    print ('%s done'%(fam))

print (nucs_count)
