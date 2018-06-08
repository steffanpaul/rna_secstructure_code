from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sb
import time as time
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

#sequence generator function
def seq_generator(num_data, seq_length, dims, seed):
    
    np.random.seed(seed)
    Xsim = np.zeros((num_data, seq_length, 1, 4), np.float32)
    for d in range(num_data):
        Xsim_key = np.random.choice([0,1,2,3], seq_length, [0.25, 0.25, 0.25, 0.25])
        Xsim_hp = np.zeros((seq_length,1, 4))
        for (idx,nuc) in enumerate(Xsim_key):
            Xsim_hp[idx][0][nuc] = 1
        Xsim[d] = Xsim_hp
    return Xsim

    #First order mutagenesis function              
def mutate(sequence, seq_length, dims):
    import numpy as np
    num_mutations = seq_length * dims
    hotplot_mutations = np.zeros((num_mutations,seq_length,1,dims)) 

    for position in range(seq_length):
        for nuc in range(dims):
            mut_seq = np.copy(sequence)          
            mut_seq[0, position, 0, :] = np.zeros(dims)
            mut_seq[0, position, 0, nuc] = 1.0
            
            hotplot_mutations[(position*dims)+nuc] = mut_seq
    return hotplot_mutations



#def secondorder_mutate(X):
def double_mutate(sequence, seq_length, dims):
    import numpy as np
    num_mutations = (seq_length * dims)*((seq_length - 1) * dims)
    mutations_matrix = np.zeros((seq_length,seq_length, dims*dims, seq_length,1,dims)) 

    for position1 in range(seq_length):
        
        for position2 in range(seq_length):
            
            for nuc1 in range(dims):
                
                for nuc2 in range(dims):
                    
                    mut_seq = np.copy(sequence)
                    mut_seq[0, position1, 0, :] = np.zeros(dims)
                    mut_seq[0, position1, 0, nuc1] = 1.0
                    mut_seq[0, position2, 0, :] = np.zeros(dims)
                    mut_seq[0, position2, 0, nuc2] = 1.0

                    mutations_matrix[position1, position2, (nuc1*dims)+nuc2, :] = mut_seq

    return mutations_matrix



'--------------------------------------------------------------------------------------------------------------------------------'

''' ANALYSIS '''



def fom_saliency(X, layer, alphabet, nntrainer, sess, title='notitle', figsize=(15,2)):

    ''' requires that deepomics is being used and the appropriate architecture has already been constructed
    Must first initialize the session and set best parameters

    layer is the activation layer we want to use as a string
    figsize is the figure size we want to use'''

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T
    
    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - predictions[0]
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    plt.figure(figsize=figsize)
    if title != 'notitle':
        plt.title(title)
    visualize.plot_seq_pos_saliency(np.squeeze(X).T, 
                                        norm_heat_mut,
                                        alphabet=alphabet, 
                                        nt_width=400) 
    
def fom_neursal(X, layer, alphabet, neuron, nntrainer, sess, title='notitle', figsize=(15,2)):
    
    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get the neurons score for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)[:,neuron]

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    dense = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T
    
    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - dense[:, neuron]
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    plt.figure(figsize=figsize)
    if title != 'notitle':
        plt.title(title)
    visualize.plot_seq_pos_saliency(np.squeeze(X).T, 
                                        norm_heat_mut,
                                        alphabet=alphabet, 
                                        nt_width=400) 
    
    
def som_average(X, savepath, nntrainer, sess, progress='on'):

    num_summary, seqlen, _, dims = X.shape

    sum_mut2_scores = np.zeros((seqlen*seqlen*dims*dims, 1))
    starttime = time.time()

    for ii in range(num_summary):
        print (ii)
        epoch_starttime = time.time()

        #mutate the sequence
        X_mutsecorder = double_mutate(np.expand_dims(X[ii], axis=0), seqlen, dims)

        #reshape the 6D tensor into a 4D tensor that the model can test
        X_mutsecorder_reshape = np.reshape(X_mutsecorder, (seqlen*seqlen*dims*dims, seqlen, 1, dims))
        mutations = {'inputs': X_mutsecorder_reshape, 'targets': np.ones((X_mutsecorder_reshape.shape[0], 1))}

        #Get output activations for the mutations
        mut2_scores= nntrainer.get_activations(sess, mutations, layer='output')

        #Sum all the scores into a single matrix
        sum_mut2_scores += mut2_scores

        epoch_endtime = time.time()
        
        if progress == 'on':

            print ('Epoch duration =' + sectotime(epoch_endtime -epoch_starttime))
            print ('Cumulative duration =' + sectotime(epoch_endtime - starttime))
            print ()
            
    if progress == 'off':
        print ('----------------Summing complete----------------')
        
    # Save the summed array for future use
    np.save(savepath, sum_mut2_scores)
    print ('Saving scores to ' + savepath)

    return (sum_mut2_scores)





def square_holplot(mutations, num, alphabet, limits=(0., 1.0), cmap ='Blues', figsize=(15,14), lines=True, start=(4,22)):

    if alphabet == 'rna':
        nuc = ['A', 'C', 'G', 'U']
    if alphabet == 'dna':
        nuc = ['A', 'C', 'G', 'T']

    if lines == True:
        linewidths = 0.1
    else:
        linewidths = 0.
        
    if limits == False:
        vmin, vmax = (None, None)
    else:
        vmin, vmax = limits

    start_1, start_2 = start

    fig = plt.figure(figsize=(15,14))
    for one in range(num):
        for two in range(num):
            ax = fig.add_subplot(num, num, ((one*num)+two)+1)

            #plot the 0th column with row labels and the num_th most row with column labels
            if two == 0:
                if one == (num-1):
                    xtick=nuc
                    ytick=nuc
                else:
                    xtick=[]
                    ytick=nuc
            else:
                if one == (num-1):
                    xtick=nuc
                    ytick=[]
                else:
                    xtick=[]
                    ytick=[]

            ax = sb.heatmap(mutations[one+start_1, two+start_2], vmin=vmin, vmax=vmax, cmap=cmap, linewidths=linewidths, linecolor='black', xticklabels=xtick, yticklabels=ytick, cbar=False)
            
            
            
def symlinear_holplot(mutations, figplot, alphabet, start=0, limits=(0., 1.), cmap ='Blues', figsize=(10,7), lines=True):

    if alphabet == 'rna':
        nuc = ['A', 'C', 'G', 'U']
    if alphabet == 'dna':
        nuc = ['A', 'C', 'G', 'T']
        
    row, col = figplot
    end = start + 2*row*col
    
    if lines == True:
        linewidths = 0.1
    if lines == False:
        linewidths = 0.
    
    if limits == False:
        vmin, vmax = (None, None)
    else:
        vmin, vmax = limits

    fig = plt.figure(figsize=figsize)
    for ii in range(row*col):
        ax = fig.add_subplot(row,col,ii+1)
        ax.set_title(str(ii)+','+str(end-ii))
        ax = sb.heatmap(mutations[ii, end-ii], vmin=vmin, vmax=vmax, cmap='Blues', linewidths=linewidths, linecolor='black', xticklabels=nuc, yticklabels=nuc)




'--------------------------------------------------------------------------------------------------------------------------------'

''' UTILITIES '''

def sectotime(t):
    t = np.around(t, 2)
    if t>=3600.:
        s = t%60
        m = ((t)%3600)//60
        hr = t//3600
        output = str(int(hr)) + 'hr ' + str(int(m)) + 'min ' + str(s) + 's'
    else:
        if t>=60.:
            s = t%60
            m = t//60
            output = str(int(m)) + 'min ' + str(s) + 's'
        else:
            s = np.copy(t)
            output = str(s) + 's'
    return(output)







































