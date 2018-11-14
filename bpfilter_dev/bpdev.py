from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import sys
sys.path.append('../../..')
import mutagenesisfunctions as mf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency
import time as time
import pandas as pd

def bp_coords(ugSS):
    '''
    Function that takes in an ungapped Sequence string and
    outputs a list of lists with the coordinates base pairs.
    Optionally it can also output the list extended with the 
    reflections of the coordinates for use with holistics 
    plots.
    '''

    bp_openers = ['(', '<', '{']
    bp_closers = [')', '>', '}']

    basepairs = [] #list to hold the base pair coords
    opened = np.array([]) # holds the integers of chars and keeps track of how close they are to being closed
    counter = 0
    for char in ugSS:

        if char in bp_openers:
            #open a base pair and start counting till its closed
            opened = np.append(opened, 0)
            opened += 1

        elif char in bp_closers: 
            #get closer to closing if we find a closing bracket
            opened -= 1
            if 0 in opened:
                #check if we've successfuly closed a pair
                op = np.where(opened ==0)[0][0]
                basepairs.append([op, counter]) #add the pair to our list
                opened[np.where(opened ==0)] = 1000 # make the recently closed char negligible
            opened = np.append(opened, 1000) #treat closing brackets as negligible


        else:
            opened = np.append(opened, 1000) #non-base-paired chars are negligible

        counter += 1
    
    basepairs = np.asarray(basepairs)
    
    #Optional reflection
    reflect = basepairs[:, ::-1]
    basepairs = np.vstack([basepairs, reflect])
    
    return (basepairs)
    
    
def KLD(hol, ref):
    S = np.ravel(hol)
    R = np.ravel(ref)
    dkl = np.sum([S[i]*(np.log(S[i]+1e-15)-np.log(R[i]+1e-15)) for i in range(len(S))])
    return (dkl)

def KLD_hol(hol_mut, ref):
    KLD_scores = np.zeros((hol_mut.shape[0], hol_mut.shape[0]))
    for one in range(hol_mut.shape[0]):
        for two in range(hol_mut.shape[0]):
            KLD_scores[one, two] = KLD(makeprob(hol_mut[one, two]), ref)
    return (KLD_scores)

def makeprob(hol):
    norm = np.sum(np.abs(hol))
    return (hol/norm)

def bp_probmx():
	bpfilter = np.ones((4,4))*0
	for i,j in zip(range(4), range(4)):
	    bpfilter[i, -(j+1)] = 0.25
	return (bpfilter)


def avgholbp(ugSS, numbp, dims, meanhol_mut2):
    #Get the base pair coords
    bp_rc = bp_coords(ugSS)

    #pull out the base pairs from the holistic scores array
    bp_hols = np.zeros((numbp, dims, dims))

    for i,r in enumerate(bp_rc):
        bp_hols[i] = meanhol_mut2[r[0],r[1]]

        bp_hols_avg = np.mean(bp_hols, axis=0)
    plt.figure()
    sb.heatmap(bp_hols_avg)
    plt.show()
    return (bp_hols_avg)




def plotKLDscores_toy(meanhol_mut2, cmap='RdPu'):
	K = KLD_hol(meanhol_mut2, bp_probmx())

	plt.figure(figsize=(15,6))
	plt.subplot(1,2,1)
	sb.heatmap(K, cmap=cmap, linewidth=0.0)
	plt.title('Base Pair score for the ungapped consensus regions given by infernal')
	plt.xlabel('Ungapped nucleotides: pos 1')
	plt.ylabel('Ungapped nucleotides: pos 2')
	plt.show()



def plotKLDscores(meanhol_mut2, bpugSQ, bpSS, cmap='RdPu'):
	K = KLD_hol(meanhol_mut2, bp_probmx())

	plt.figure(figsize=(15,6))
	plt.subplot(1,2,1)
	sb.heatmap(K, xticklabels=bpugSQ, yticklabels=bpugSQ, vmax=None, cmap=cmap, linewidth=0.0)
	plt.title('Base Pair score for the ungapped consensus regions given by infernal')
	plt.xlabel('Ungapped nucleotides: pos 1')
	plt.ylabel('Ungapped nucleotides: pos 2')
	plt.show()
	plt.subplot(1,2,2)
	sb.heatmap(K[bpugidx][:, bpugidx], xticklabels=bpSS, yticklabels=bpSS, vmax=None, cmap=cmap, linewidth=0.)
	plt.title('Base Pair score for the base paired consensus regions given by infernal')
	plt.xlabel('Base Paired nucleotides: pos 1')
	plt.ylabel('Base Paired nucleotides: pos 2')
	plt.show

