from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import numpy as np
import tensorflow as tf
#import helper
#from helper import BatchGenerator

def g_test(X, batch_size=1024):
    def calculate_g_test(X, batch_size):
        eps = 1e-7 # small number to prevent indefinite/infinite values
        N, seq_length, num_alphabet = X.shape
        G = np.zeros((seq_length, seq_length))

        bg = BatchGenerator(N, batch_size, shuffle=False)

        for i in range(seq_length):
            for j in range(i):
                resid_i = X[:,i,1:]
                resid_j = X[:,j,1:]

                # find pairs without gaps
                O_ij = 0
                for index in bg.indices:
                    O_ij += np.dot(np.transpose(X[index,i,1:]), X[index,j,1:])

                index = np.where(np.sum(resid_i + resid_j, axis=1) == 2)[0]
                N_ij = len(index)
                O_ij /= N_ij

                # individual frequencies
                O_i = np.sum(O_ij, axis=0, keepdims=True) + eps
                O_j = np.sum(O_ij, axis=1, keepdims=True) + eps

                G[i,j] = np.sum(O_ij*np.log(O_ij/(O_i*O_j) + eps))

        # fill out upper symmetric triangle
        G += G.T

        return clean_diagonal_cmap(G)

    if isinstance(X, (list, tuple)):
        num_samples = len(X)
        mi_list = []
        for i in range(num_samples):
            print('G-test on sample %d out of %d'%(i+1, num_samples))
            mi_list.append(clean_diagonal_cmap(calculate_g_test(X[i], batch_size)))
        return mi_list
    else:
        return clean_diagonal_cmap(calculate_g_test(X, batch_size))

def mutual_information(X, batch_size=1024):

    def clean_diagonal_cmap(M):
        seq_length = len(M)
        mu = np.mean(M[np.triu_indices(seq_length, k=2)])
        for i in range(seq_length):
            M[i,i] = mu
        return M

    def calculate_mi(X, batch_size):
        eps = 1e-7 # small number to prevent indefinite/infinite values
        N, seq_length, num_alphabet = X.shape
        M = np.zeros((seq_length, seq_length))

        bg = BatchGenerator(N, batch_size, shuffle=False)

        # loop over first position
        for i in range(seq_length):

            # loop over second position
            for j in range(i):
                resid_i = X[:,i,:]
                resid_j = X[:,j,:]
                #f_ij = np.dot(np.transpose(resid_i), resid_j)/N
                # calculate number of co-occurences of AA
                f_ij = 0
                for index in bg.indices:
                    f_ij += np.dot(np.transpose(resid_i[index]), resid_j[index])
                f_ij /= N

                f_i = np.sum(f_ij, axis=0, keepdims=True) + 1e-7
                f_j = np.sum(f_ij, axis=1, keepdims=True) + 1e-7
                M[i,j] = np.sum(f_ij*np.log(f_ij/(f_i*f_j) + 1e-7))

        # fill out upper symmetric triangle
        M += M.T
        return clean_diagonal_cmap(M)

    if isinstance(X, (list, tuple)):
        num_samples = len(X)
        mi_list = []
        for i in range(num_samples):
            print('MI on sample %d out of %d'%(i+1, num_samples))
            mi_list.append(clean_diagonal_cmap(calculate_mi(X[i], batch_size)))
        return mi_list
    else:
        return clean_diagonal_cmap(calculate_mi(X, batch_size))


def clean_diagonal_cmap(M):
    seq_length = len(M)
    mu = np.mean(M[np.triu_indices(seq_length, k=2)])
    for i in range(seq_length):
        M[i,i] = mu
    return M

def apc_correction(M):

    def fix_apc(M):
        apc = np.mean(M, axis=0, keepdims=True)*np.mean(M, axis=1, keepdims=True)/np.mean(M)
        M_apc = M - apc
        M_apc = clean_diagonal_cmap(M_apc)
        return M_apc
    if isinstance(M, (list, tuple)):
        M_apc = []
        for i in range(len(M)):
            M_apc.append(fix_apc(M[i]))
        return M_apc
    else:
        return fix_apc(M)



#-----------------------------------------------------------------------------------
# Batch Generator class
#-----------------------------------------------------------------------------------

class BatchGenerator():
    """ helper class to generate mini-batches """

    def __init__(self, num_data, batch_size=128, shuffle=False):

        self.num_data = num_data
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_data/batch_size))
        self.batch_index = 0
        self.generate_batches(batch_size, shuffle)

    def generate_batches(self, batch_size=128, shuffle=False):

        self.num_batches = int(np.ceil(self.num_data/batch_size))
        self.batch_index = 0        # reset batch_index

        if shuffle == True:
            index = np.random.permutation(self.num_data)
        else:
            index = range(self.num_data)

        self.indices = []
        for i in range(self.num_batches):
            self.indices.append(index[i*batch_size:i*batch_size+batch_size])

        # get remainder
        index = range(self.num_batches*batch_size, self.num_data)
        if index:
            self.indices.append(index)


    def next_batch(self, data):
        """Generate next mini-batch of data"""
        indices = np.sort(self.indices[self.batch_index])

        if isinstance(data, (list, tuple)):
            data_batch = []
            for vals in data:
                data_batch.append(vals[indices])
        else:
            data_batch = data[indices]

        self.batch_index += 1
        if self.batch_index == self.num_batches:
            self.batch_index = 0

        return data_batch

    def get_batch_index(self):
        return self.batch_index

    def get_num_batches(self):
        return self.num_batches
