from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import scipy

from sklearn.metrics import roc_curve, auc, precision_recall_curve

import sys
sys.path.append('../../..')
import mutagenesisfunctions as mf
import helper 

from Bio import AlignIO
import time as time
import pandas as pd

def make_variable_length(X_train_fixed, addon=5):
    N = len(X_train_fixed)
    start_buffer = np.random.randint(1, addon, size=N)
    end_buffer = np.random.randint(1, addon, size=N)

    X_train_all = []
    starts = []
    for n in range(N):
        start_size = start_buffer[n]
        start_index = np.random.randint(0,4, size=start_size)
        start_one_hot = np.zeros((start_size, 4))
        for i in range(start_size):
            start_one_hot[i, start_index[i]] = 1
        starts.append(len(start_index))
        
        end_size = end_buffer[n]
        end_index = np.random.randint(0,4, size=end_size)
        end_one_hot = np.zeros((end_size, 4))
        for i in range(end_size):
            end_one_hot[i, end_index[i]] = 1
        X_train_all.append(np.vstack([start_one_hot, X_train_fixed[n], end_one_hot]))
    return X_train_all, starts
#---------------------------------------------------------------------------------------------------------------------------------

'''DEFINE ACTIONS'''
TEST = False
WRITE = False
FOM = False
SOMCALC = False
SOMVIS = False

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

'''DEFINE LOOP'''
trials = ['med']#['small', 'med', 'large']
varlengths = [10, 20, 30] #20
exp = 'toyhp'  #for both the data folder and the params folder
exp_data = 'data_%s'%(exp)

img_folder = 'Images'

for t in trials:
    for v in varlengths:
    #---------------------------------------------------------------------------------------------------------------------------------

        '''OPEN DATA'''

        starttime = time.time()

        #Open data from h5py
        filename = '%s_50k_%s.hdf5'%(exp, t)
        data_path = os.path.join('../..', exp_data, filename)
        with h5py.File(data_path, 'r') as dataset:
            X_data = np.array(dataset['X_data'])
            Y_data = np.array(dataset['Y_data'])
            
        numdata, seqlen, dims = X_data.shape


        #Make variable!
        addon = v
        X_data, starts = make_variable_length(X_data, addon=addon)

        # get validation and test set from training set
        test_frac = 0.3
        valid_frac = 0.1
        N = numdata
        split_1 = int(N*(1-valid_frac-test_frac))
        split_2 = int(N*(1-test_frac))
        shuffle = np.random.permutation(N)

        X_train = [X_data[s] for s in shuffle[:split_1]]
        X_valid = [X_data[s] for s in shuffle[split_1:split_2]]
        X_test = [X_data[s] for s in shuffle[split_2:]]
        test_starts = np.asarray(starts)[shuffle[split_2:]]

        Y_train = Y_data[shuffle[:split_1]]
        Y_valid = Y_data[shuffle[split_1:split_2]]
        Y_test = Y_data[shuffle[split_2:]]
          
        print ('Data extraction and dict construction completed in: ' + mf.sectotime(time.time() - starttime))

        #---------------------------------------------------------------------------------------------------------------------------------

        '''BUILD MODEL AND OPTIMIZER'''

        tf.reset_default_graph()

        num_hidden = 64
        num_layers = 2
        num_classes = Y_train.shape[1]

        # tf Graph input
        X = tf.placeholder(tf.float32, [None, None, X_train[0].shape[1]], name='inputs')
        Y = tf.placeholder(tf.float32, [None, num_classes], name='ouputs')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        lstm1_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)#, forget_bias=1.0)
        lstm1_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm1_fw_cell, 
                                                 output_keep_prob=keep_prob, 
                                                 state_keep_prob=1.0,
                                                 variational_recurrent=False,
                                                 dtype=tf.float32)

        lstm1_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)#, forget_bias=1.0)
        lstm1_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm1_bw_cell, 
                                                 output_keep_prob=keep_prob, 
                                                 state_keep_prob=1.0,
                                                 variational_recurrent=False,
                                                 dtype=tf.float32)

        outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(lstm1_fw_cell, lstm1_bw_cell, X, 
                                                           sequence_length=helper.length(X), dtype=tf.float32,
                                                           scope='BLSTM_1')

        outputs_forward, outputs_backward = outputs1

        # states_forward is a tuple of (c is the hidden state and h is the output)
        concat_outputs = tf.concat([outputs_forward, outputs_backward], axis=2, name='intermediate')

        lstm2_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)#, forget_bias=1.0)
        lstm2_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm2_fw_cell, 
                                                 output_keep_prob=keep_prob, 
                                                 state_keep_prob=1.0,
                                                 variational_recurrent=False,
                                                 dtype=tf.float32)

        lstm2_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)#, forget_bias=1.0)
        lstm2_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm2_bw_cell, 
                                                 output_keep_prob=keep_prob, 
                                                 state_keep_prob=1.0,
                                                 variational_recurrent=False,
                                                 dtype=tf.float32)

        outputs2, states2 = tf.nn.bidirectional_dynamic_rnn(lstm2_fw_cell, lstm2_bw_cell, concat_outputs,
                                                            scope='BLSTM_2', dtype=tf.float32)

        states_forward, states_backward = states2

        # states_forward is a tuple of (c is the hidden state and h is the output)
        concat_states = tf.concat([states_forward[1], states_backward[1]], axis=1, name='output')

        # Linear activation, using rnn inner loop last output
        W_out = tf.Variable(tf.random_normal([num_hidden*2, num_classes]))
        b_out = tf.Variable(tf.random_normal([num_classes]))

        #last = tf.gather(outputs, int(outputs.get_shape()[1])-1)  
        #last = int(outputs.get_shape()[1]) - 1
        logits = tf.matmul(concat_states, W_out) + b_out
        predictions = tf.nn.sigmoid(logits)

        # The Optimizer

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Define loss and optimizer
        predictions = tf.clip_by_value(predictions, clip_value_max=1-1e-7, clip_value_min=1e-7)
        #cost = tf.reduce_sum(Y*tf.log(predictions), axis=1)
        cost = tf.reduce_sum(Y*tf.log(predictions)+(1-Y)*tf.log(1-predictions), axis=1)

        total_loss = tf.reduce_mean(-cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = optimizer.apply_gradients(grads)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # Evaluate model (with test logits, for dropout to be disabled)
        #correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
        correct_pred = tf.equal(tf.round(predictions), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        #---------------------------------------------------------------------------------------------


        '''SAVE PATHS AND PARAMETERS'''
        params_results = '../../results'

        modelarch = 'rnn'
        trial = 'var' + str(v) + t
        modelsavename = '%s_%s'%(modelarch, trial)

        '''LOAD PARAMETERS'''
        save_path = os.path.join(params_results, exp)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            print("making directory: " + save_path)
        params_filename = '%s_%s_best'%(modelarch, trial)
        params_path = os.path.join(save_path, params_filename)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # restore trained parameters
        saver = tf.train.Saver()
        saver.restore(sess, save_path=params_path)
        print ('Restoring parameters from: %s'%(params_path))

#---------------------------------------------------------------------------------------------
        '''TEST'''
        batch_size = 128
        batches, sort_index = helper.bucket_generator(X_test, Y_test, batch_size, index=True)
        num_batches = len(batches)

        loss = 0
        acc = 0
        valid_predictions = []
        valid_truth = []
        start_time = time.time()
        num_batches = len(batches)
        bar_length = 25

        for i, batch in enumerate(batches):

            batch_loss, batch_predict, batch_logits = sess.run([total_loss, predictions, logits], feed_dict={X: batch[0], 
                                                                                    Y: batch[1], 
                                                                                    keep_prob: 1.0})            
            loss += batch_loss
            valid_predictions.append(batch_predict)
            valid_truth.append(batch[1])
            
            remaining_time = (time.time()-start_time)*(num_batches-(i+1))/(i+1)
            percent = float(i)/num_batches
            progress = '='*int(round(percent*bar_length))
            spaces = ' '*int(bar_length-round(percent*bar_length))
            sys.stdout.write("\r[%s] %.1f%% -- remaining time=%.2fs -- loss=%.5f -- acc=%.5f" \
            %(progress+spaces, percent*100, remaining_time, loss/(i+1), acc/(i+1)))

        sys.stdout.write("\r[%s] %.1f%% -- elapsed time=%.2fs -- loss=%.5f -- acc=%.5f\n" \
        %(progress+spaces, percent*100, time.time()-start_time, loss/(i+1), acc/(i+1)))
        sys.stdout.write("\n")

        valid_predictions = np.vstack(valid_predictions)
        valid_truth = np.vstack(valid_truth)

        correct = np.mean(np.equal(valid_truth, np.round(valid_predictions)))
        auc_roc, roc_curves = helper.roc(valid_truth, valid_predictions)
        auc_pr, pr_curves = helper.pr(valid_truth, valid_predictions)
        mean_vals = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

        print("  test loss  = "+str(loss/num_batches))
        print("  test acc   = "+str(np.nanmean(correct)))
        print("  test AUROC = "+str(np.nanmean(auc_roc)))
        print("  test AUPRC = "+str(np.nanmean(auc_pr)))
        
        if WRITE:
            metricsline = '%s,%s,%s,%s,%s,%s,%s'%(exp, modelarch, trial, loss, mean_vals[0], mean_vals[1], mean_vals[2])
            fd = open('test_metrics.csv', 'a')
            fd.write(metricsline+'\n')
            fd.close()



        #---------------------------------------------------------------------------------------------
        '''SORT ACTIVATIONS'''
        WT_predictions = valid_predictions[np.argsort(sort_index)]
        plot_index = np.argsort(WT_predictions[:,0])[::-1]

        #---------------------------------------------------------------------------------------------
        '''SECOND ORDER MUTAGENESIS'''

        '''Som Calc'''
        if SOMCALC:
            def second_order_mutagenesis(sess, predictions, X_val, ugidx):
                seqlen, dims = X_val.shape
                idxlen = len(ugidx)

                # get wild-type score
                wt_score = sess.run(predictions, feed_dict={X: np.expand_dims(X_val, axis=0), keep_prob: 1.0})[0]     

                # generate mutagenesis sequences
                num_mut = idxlen**2*dims**2
                X_mut = np.einsum('nl,lka->nka', np.ones((num_mut, 1)), np.expand_dims(X_val, axis=0))

                k=0
                for position1 in ugidx:
                    for position2 in ugidx:
                        for nuc1 in range(dims):
                            for nuc2 in range(dims):
                                X_mut[k, position1, :] = 0
                                X_mut[k, position1, nuc1] = 1        
                                X_mut[k, position2, :] = 0
                                X_mut[k, position2, nuc2] = 1        
                                k += 1

                # get second order mutagenesis score
                X_mut = [x for x in X_mut] #convert to lists because this is what the RNN code is used to
                mut_scores = []
                batches = helper.batch_generator(X_mut, batch_size=512, MAX=None, shuffle_data=False)
                for i, batch in enumerate(batches):
                    batch_predict = sess.run(predictions, feed_dict={X: batch, keep_prob: 1.0})            
                    mut_scores.append(batch_predict)
                mut_scores = np.vstack(mut_scores)
                
                # calculate log-odds score
                log_odds = np.log(mut_scores + 1e-7) - np.log(wt_score + 1e-7)


                # reshape second order scores
                second_mutagenesis_logodds = np.zeros((idxlen, idxlen, dims, dims))
                k = 0
                for i in range(idxlen):
                    for j in range(idxlen):
                        for m in range(dims):
                            for n in range(dims):
                                second_mutagenesis_logodds[i,j,m,n] = log_odds[k,0]
                                k += 1
                return second_mutagenesis_logodds


            start_time = time.time()
            np.random.seed(274)

            bar_length = 50
            N = 2000
            mutagenesis_logodds = []
            for i, index in enumerate(plot_index[:N]):
                ugidx = range(test_starts[index], seqlen + test_starts[index])
                logresult = second_order_mutagenesis(sess, logits, X_test[index], ugidx)
                mutagenesis_logodds.append(logresult)


                remaining_time = (time.time()-start_time)*(N-(i+1))/(i+1)
                percent = float(i)/N
                progress = '='*int(round(percent*bar_length))
                spaces = ' '*int(bar_length-round(percent*bar_length))
                sys.stdout.write("\r[%s] %.1f%% -- remaining time=%.2fs" \
                %(progress+spaces, percent*100, remaining_time))

            sys.stdout.write("\r[%s] %.1f%% -- elapsed time=%.2fs" \
            %(progress+spaces, percent*100, time.time()-start_time))
            sys.stdout.write("\n")

            mean_mut2 = np.nanmean(mutagenesis_logodds, axis=0)
            idx = np.where(np.isnan(mean_mut2))
            mean_mut2[idx] = np.min(mean_mut2)

            arrayspath = 'Arrays/%s_%s%s_so.npy'%(exp, modelarch, trial)
            np.save(arrayspath, mean_mut2)

        #---------------------------------------------------------------------------------------------
        '''Som Vis'''

        if SOMVIS:
            arrayspath = 'Arrays/%s_%s%s_so.npy'%(exp, modelarch, trial)

            mean_mut2 = np.load(arrayspath)

            norm_mean_mut2 = helper.normalize_mut_hol(mean_mut2, normfactor=1)

            #Let's try something weird
            bpfilter = np.ones((4,4))*-1
            for i,j in zip(range(4), range(4)):
                bpfilter[i, -(j+1)] = 1.

            C = np.sum((norm_mean_mut2*bpfilter).reshape(seqlen,seqlen,dims*dims), axis=2)
            #C = C - np.mean(C)
            #C = C/np.max(C)

            color = 'Oranges'

            plt.figure(figsize=(8,6))
            sb.heatmap(C, vmin=None, cmap=color , linewidth=0.00)
            plt.title('Base Pair scores: %s %s %s'%(exp, modelarch, trial))

            som_file = modelsavename + 'SoM_bpfilter' + '.png'
            som_file = os.path.join(img_folder, som_file)
            plt.savefig(som_file)
            plt.close()

            blocklen = np.sqrt(np.product(mean_mut2.shape)).astype(int)
            S = np.zeros((blocklen, blocklen))
            i,j,k,l = mean_mut2.shape

            for ii in range(i):
                for jj in range(j):
                    for kk in range(k):
                        for ll in range(l):
                            S[(4*ii)+kk, (4*jj)+ll] = mean_mut2[ii,jj,kk,ll]

            plt.figure(figsize=(15,15))
            plt.imshow(S,  cmap='RdPu')
            plt.colorbar()
            plt.title('Blockvis of all mutations: %s %s %s'%(exp, modelarch, trial))

            som_file =modelsavename+ 'SoM_blockvis' + '.png'
            som_file = os.path.join(img_folder, som_file)
            plt.savefig(som_file)
            plt.close()
        #---------------------------------------------------------------------------------------------
