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

'''DEFINE ACTIONS'''
TRAIN = False
WRITE = False
FOM = False
SOMCALC = False
SOMVIS = False

if '--train' in sys.argv:
    TRAIN = True
if '--write' in sys.argv:
  WRITE = True
if '--fom' in sys.argv:
  FOM = True
if '--somcalc' in sys.argv:
  SOMCALC = True
if '--somvis' in sys.argv:
  SOMVIS = True

#---------------------------------------------------------------------------------------------------------------------------------
'''DEFINE LOOP'''
trials = ['trna', 'riboswitch', 'glna']
#trainportion_list = [0.7, 0.5, 0.3, 0.1] #0.7 is the original trainportion we've been working with
#trainportion_list = [0.08, 0.06, 0.04, 0.02, 0.01] 
trainportion_list = [0.7, 0.5, 0.3, 0.1, 0.08, 0.06]#, 0.04, 0.02, 0.01, 0.005, 0.001]

#The first part of this dictionary is an artifact from a previous script
datafiles = {'glna': ['glna_100k_d8.hdf5', '../../data_RFAM/glnAsim_100k.sto'], 
              'trna': ['trna_100k_d4.hdf5', '../../data_RFAM/trnasim_100k.sto'],
              'riboswitch': ['riboswitch_100k_d4.hdf5', '../../data_RFAM/riboswitch_100k.sto'],}

exp = 'trainsize'  #for the params folder
modelarch = 'rnn'


img_folder = 'Images'

for t in trials:
  for trainportion in trainportion_list:

    #---------------------------------------------------------------------------------------------------------------------------------

        '''OPEN DATA'''

        starttime = time.time()

        #Open data from h5py
        exp_data = 'data_background'
        filename = '%s_100k_sh%.0f.hdf5'%(t, 0.25*100)
        data_path = os.path.join('../..', exp_data, filename)
        with h5py.File(data_path, 'r') as dataset:
            X_data = np.array(dataset['X_data'])
            Y_data = np.array(dataset['Y_data'])
            
        numdata, seqlen, _, dims = X_data.shape
        dims = dims-1

        #remove gaps from sequences
        ungapped = True
        if ungapped:
            X_data = X_data[:, :, :, :dims]
            
        # get validation and test set from training set
        train_frac = trainportion
        valid_frac = 0.1
        test_frac = 1-trainportion-valid_frac
        N = numdata
        split_1 = int(N*(1-valid_frac-test_frac))
        split_2 = int(N*(1-test_frac))
        shuffle = np.random.permutation(N)

        X_train = X_data[shuffle[:split_1], :, 0, :]
        X_valid = X_data[shuffle[split_1:split_2], :, 0, :]
        X_test = X_data[shuffle[split_2:], :, 0, :]

        Y_train = Y_data[shuffle[:split_1]]
        Y_valid = Y_data[shuffle[split_1:split_2]]
        Y_test = Y_data[shuffle[split_2:]]
            
        print ('Data extraction and dict construction completed in: ' + mf.sectotime(time.time() - starttime))
          
        #Get the full secondary structure and sequence consensus from the emission
        simalign_file = datafiles[t][1]
        SS = mf.getSSconsensus(simalign_file)
        SQ = mf.getSQconsensus(simalign_file)

        #Get the ungapped sequence and the indices of ungapped nucleotides
        _, ugSS, ugidx = mf.rm_consensus_gaps(X_data, SS)
        _, ugSQ, _ = mf.rm_consensus_gaps(X_data, SQ)


        #Get the sequence and indices of the conserved base pairs
        bpchars = ['(',')','<','>','{','}']
        sig_bpchars = ['<','>']
        bpidx, bpSS, nonbpidx = mf.sigbasepair(SS, bpchars)
        numbp = len(bpidx)
        numug = len(ugidx)

        #Get the bpug information
        bpugSQ, bpugidx = mf.bpug(ugidx, bpidx, SQ)

        #Unalign and make the sequences variable for the RNN
        def unalign(X):
            nuc_index = np.where(np.sum(X, axis=1)!=0)
            return (X[nuc_index])

        X_train_unalign = [unalign(X) for X in X_train]
        X_valid_unalign = [unalign(X) for X in X_valid]
        X_test_unalign = [unalign(X) for X in X_test]

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

        if TRAIN:
            '''SAVE PATHS AND PARAMETERS'''
            params_results = '../../results'

            #Pick the appropriate number of zeros to get our names the same space.
            tp = trainportion
            if tp>= 0.1:
                trial = t + '_tp0%.0f'%(tp*100)
            elif tp>= 0.01:
                trial = t + '_tp00%.0f'%(tp*100)
            elif tp>= 0.001:
                trial = t + '_tp00%.0f'%(tp*1000)

            modelsavename = '%s_%s'%(modelarch, trial)

            '''TRAIN MODEL'''

            # start session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            batch_size = 128

            train_batches = helper.bucket_generator(X_train_unalign, Y_train, batch_size)
            valid_batches = helper.bucket_generator(X_valid_unalign, Y_valid, batch_size)

            num_epochs = 75
            bar_length = 25
            patience = 10


            # path to save results
            save_path = os.path.join(params_results, exp)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
                print("making directory: " + save_path)
            params_filename = '%s_%s_best'%(modelarch, trial)
            params_path = os.path.join(save_path, params_filename)
                
            wait=0
            min_loss = 1e10
            for epoch in range(num_epochs):
                print('epoch: '+ str(epoch+1))
                
                num_batches = len(train_batches)
                shuffled_batches = []
                for i in np.random.permutation(num_batches):
                    shuffled_batches.append(train_batches[i])
                    
                loss = 0
                acc = 0
                start_time = time.time()
                for i, batch in enumerate(shuffled_batches):
                    batch_loss, batch_acc, _ = sess.run([total_loss, accuracy, train_op], feed_dict={X: batch[0], 
                                                                                                  Y: batch[1], 
                                                                                                  keep_prob: 0.5,
                                                                                                  learning_rate: 0.0003})            
                    loss += batch_loss
                    acc += batch_acc

                    remaining_time = (time.time()-start_time)*(num_batches-(i+1))/(i+1)
                    percent = float(i)/num_batches
                    progress = '='*int(round(percent*bar_length))
                    spaces = ' '*int(bar_length-round(percent*bar_length))
                    sys.stdout.write("\r[%s] %.1f%% -- remaining time=%.2fs -- loss=%.5f -- acc=%.5f" \
                    %(progress+spaces, percent*100, remaining_time, loss/(i+1), acc/(i+1)))
                    
                sys.stdout.write("\r[%s] %.1f%% -- elapsed time=%.2fs -- loss=%.5f -- acc=%.5f\n" \
                %(progress+spaces, percent*100, time.time()-start_time, loss/(i+1), acc/(i+1)))
                sys.stdout.write("\n")
                
                
                num_batches = len(valid_batches)
                loss = 0
                acc = 0
                valid_predictions = []
                valid_truth = []
                start_time = time.time()
                for i, batch in enumerate(valid_batches):
                    batch_loss, batch_predict = sess.run([total_loss, predictions], feed_dict={X: batch[0], 
                                                                                            Y: batch[1], 
                                                                                            keep_prob: 1.0})            
                    loss += batch_loss
                    valid_predictions.append(batch_predict)
                    valid_truth.append(batch[1])
                valid_loss = loss/num_batches
                valid_predictions = np.vstack(valid_predictions)
                valid_truth = np.vstack(valid_truth)
                
                correct = np.mean(np.equal(valid_truth, np.round(valid_predictions)))
                auc_roc, roc_curves = helper.roc(valid_truth, valid_predictions)
                auc_pr, pr_curves = helper.pr(valid_truth, valid_predictions)
                print("  valid loss  = "+str(loss/num_batches))
                print("  valid acc   = "+str(np.nanmean(correct)))
                print("  valid AUROC = "+str(np.nanmean(auc_roc)))
                print("  valid AUPRC = "+str(np.nanmean(auc_pr)))
                
                # check if current validation loss is lower, if so, save parameters, if not check patience
                if valid_loss < min_loss:
                    print("  Lower validation loss found. Saving parameters to: "+params_path)
                    
                    # save model parameters
                    saver = tf.train.Saver()
                    saver.save(sess, save_path=params_path)
                    
                    # set minimum loss to the current validation loss
                    min_loss = valid_loss
                    
                    # reset wait time
                    wait = 0
                else:
                    
                    # add to wait time
                    wait += 1
                    
                    # check to see if patience has run out
                    if wait == patience:
                        print("Patience ran out... early stopping!")
                        break

            # close tensorflow session (Note, the graph is still open)
            sess.close()

            #---------------------------------------------------------------------------------------------


        '''SAVE PATHS AND PARAMETERS'''
        params_results = '../../results'

        #Pick the appropriate number of zeros to get our names the same space.
        tp = trainportion
        if tp>= 0.1:
            trial = t + '_tp0%.0f'%(tp*100)
        elif tp>= 0.01:
            trial = t + '_tp00%.0f'%(tp*100)
        elif tp>= 0.001:
            trial = t + '_tp00%.0f'%(tp*1000)

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
        batches, sort_index = helper.bucket_generator(X_test_unalign, Y_test, batch_size, index=True)
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

            #define the SoM function
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
                X_mut = [x for x in X_mut]
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


            #Implement Second order Mutagenesis
            start_time = time.time()
            np.random.seed(274)

            bar_length = 50
            N = 2000
            mutagenesis_logodds = []
            for i, index in enumerate(plot_index[:N]):
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

            #Convert Nans to the minimum value of the array
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

            C = np.sum((norm_mean_mut2*bpfilter).reshape(numug,numug,dims*dims), axis=2)
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

        #---------------------------------------------------------------------------------------------