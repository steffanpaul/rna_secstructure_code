from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import scipy

import sys
import helptransfer as htf
sys.path.append('../../../..')
import mutagenesisfunctions as mf
import bpdev as bd
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

#from Bio import AlignIO
import time as time
import pandas as pd
#---------------------------------------------------------------------------------------------------------------------------------
'''DEFINE ACTIONS'''
TRAIN = False
WRITE = False
FOM = False
SOMCALC = False
SOMVIS = False
TRANSFER = False
SOME = False
JUSTPKHP = False
PRETRANSFER = False


if '--pretransfer' in sys.argv:
  PRETRANSFER = True
if '--transfer' in sys.argv:
  TRANSFER = True
if '--some' in sys.argv:
  SOME = True
if '--justpkhp' in sys.argv:
  JUSTPKHP = True
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

exp = 'toypk'  #for the params folder
modelarch = 'rnn'

datatype = sys.argv[1]
trialnum = sys.argv[2]
if SOME:
  portion = int(sys.argv[sys.argv.index('--some')+1]) #pulls out the divisor of the data portion eg. 50 = 50,000/50 = 1000 seqs

if '--setepochs' in sys.argv: #set the number of epochs over which the model will train (with no patience)
  numepochs = int(sys.argv[sys.argv.index('--setepochs')+1])
else:
  numepochs = 100

img_folder = 'Images_%s_d%s'%(modelarch, datatype)
if not os.path.isdir(img_folder):
  os.mkdir(img_folder)
#---------------------------------------------------------------------------------------------------------------------------------

'''OPEN DATA'''

starttime = time.time()

#Open data from h5py
exp_data = 'data_toypk'
filename = 'toypkhp_50_d%s.hdf5'%(datatype)
data_path = os.path.join('../../..', exp_data, filename)

if TRANSFER: #import pkhp data to transfer learn
    ext = '_pkhp'
elif JUSTPKHP:
    ext = '_pkhp'
elif PRETRANSFER:
    ext = '_hp'

with h5py.File(data_path, 'r') as dataset:
    X_pos = np.array(dataset['X_pos%s'%(ext)])
    X_neg = np.array(dataset['X_neg%s'%(ext)])

    Y_pos = np.array(dataset['Y_pos'])
    Y_neg = np.array(dataset['Y_neg'])


X_pos = np.expand_dims(X_pos, axis=2)
X_neg = np.expand_dims(X_neg, axis=2)

numdata, seqlen, _, dims = X_pos.shape

if not SOME:
    X_data = np.concatenate((X_pos, X_neg), axis=0)
    Y_data = np.concatenate((Y_pos, Y_neg), axis=0)
if SOME:

    X_data = np.concatenate((X_pos[:numdata//portion], X_neg[:numdata//portion]), axis=0)
    Y_data = np.concatenate((Y_pos[:numdata//portion], Y_neg[:numdata//portion]), axis=0)
# get validation and test set from training set
if not TRANSFER: #set the proportions for pretransfer
    train_frac = 0.5 #This means the pretransfer model is training on 25,000 pos and 25,000 neg sequences
    valid_frac = 0.2
    test_frac = 0.3
if TRANSFER or JUSTPKHP:
    train_frac = 0.8
    valid_frac = 0.1
    test_frac = 0.1

numdata, seqlen, _, dims = X_data.shape

N = numdata
posidx = np.random.permutation(np.arange(N//2))
negidx = np.random.permutation(np.arange(N//2, N))
split_1 = int((N//2)*(1-valid_frac-test_frac))
split_2 = int((N//2)*(1-test_frac))
#shuffle = np.random.permutation(N)

trainidx = np.random.permutation(np.concatenate([posidx[:split_1], negidx[:split_1]]))
valididx = np.random.permutation(np.concatenate([posidx[split_1:split_2], negidx[split_1:split_2]]))
testidx = np.random.permutation(np.concatenate([posidx[split_2:], negidx[split_2:]]))

X_train = X_data[trainidx, :, 0, :]
X_valid = X_data[valididx, :, 0, :]
X_test = X_data[testidx, :, 0, :]

Y_train = Y_data[trainidx]
Y_valid = Y_data[valididx]
Y_test = Y_data[testidx]

print ('Data extraction and dict construction completed in: ' + mf.sectotime(time.time() - starttime))
#---------------------------------------------------------------------------------------------------------------------------------


'''SAVE PATHS AND PARAMETERS'''
params_results = '../../../results'

trial = 'pkhp_d%st%s'%(datatype, trialnum)
if '--setepochs' in sys.argv:
  trial = 'pkhp_d%st%se%s'%(datatype, trialnum, numepochs)

if PRETRANSFER:
  trial = 'pkhp_d%s_pretran'%(datatype)
  numepochs = 5


modelsavename = '%s_%s'%(modelarch, trial)


#---------------------------------------------------------------------------------------------------------------------------------


'''BUILD MODEL AND OPTIMIZER'''

tf.reset_default_graph()



num_hidden = 64

if '--sethidden' in sys.argv:
  num_hidden = int(sys.argv[sys.argv.index('--sethidden')+1])
  trial = 'pkhp_d%st%se%sh%s'%(datatype, trialnum, numepochs, num_hidden)
  modelsavename = '%s_%s'%(modelarch, trial)

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
#SET UP A CLAUSE TO INITIATE TRANSFER LEARNING
if TRANSFER and TRAIN:
    #make the pretransfer file a copy of what we want now
    htf.import_pretransfer(params_results, exp, datatype, modelarch, modelsavename, isrnn=True)

'''TRAIN MODEL'''
if TRAIN:

  # path to save results
  save_path = os.path.join(params_results, exp)
  if not os.path.isdir(save_path):
      os.mkdir(save_path)
      print("making directory: " + save_path)
  params_filename = '%s_best'%(modelsavename)
  params_path = os.path.join(save_path, params_filename)

  # start session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  if TRANSFER:
      # restore trained parameters
      saver = tf.train.Saver()
      saver.restore(sess, save_path=params_path)
      print ('Restoring parameters from: %s'%(params_path))


  batch_size = 128
  train_batches = helper.bucket_generator(X_train, Y_train, batch_size)
  valid_batches = helper.bucket_generator(X_valid, Y_valid, batch_size)

  #numepochs = called up above
  bar_length = 25
  patience = numepochs

  wait=0
  min_loss = 1e10
  for epoch in range(numepochs):
      print('epoch: '+ str(epoch+1))


      #ITERATE OVER TRAIN SEQUENCES
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


      #ITERATE OVER VALID SEQUENCES
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


'''LOAD PARAMETERS'''
save_path = os.path.join(params_results, exp)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    print("making directory: " + save_path)
params_filename = '%s_best'%(modelsavename)
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


'''SORT ACTIVATIONS'''
WT_predictions = valid_predictions[np.argsort(sort_index)]
plot_index = np.argsort(WT_predictions[:,0])[::-1]


'''SECOND ORDER MUTAGENESIS'''
#specify how many seqs to average over in SoM
num_summary = np.min([2000, len(X_test)//2])

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


    # IMPLEMENT

    start_time = time.time()
    np.random.seed(274)

    bar_length = 50
    N = num_summary
    mutagenesis_logodds = []
    for i, index in enumerate(plot_index[:N]):
        logresult = second_order_mutagenesis(sess, logits, X_test[index], range(seqlen))
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

    #correct for negative logit scores
    mean_mut2 = np.nanmean(mutagenesis_logodds, axis=0)
    idx = np.where(np.isnan(mean_mut2))
    mean_mut2[idx] = np.min(mean_mut2)

    arrayspath = 'Arrays/%s_%s_so%.0fk.npy'%(exp, modelsavename, num_summary/1000)
    np.save(arrayspath, mean_mut2)

#---------------------------------------------------------------------------------------------
'''Som Vis'''

if SOMVIS:
    arrayspath = 'Arrays/%s_%s_so%.0fk.npy'%(exp, modelsavename, num_summary/1000)

    mean_mut2 = np.load(arrayspath)

    norm_mean_mut2 = helper.normalize_mut_hol(mean_mut2, normfactor=1)

    #Let's try something weird
    bpfilter = np.ones((4,4))*-1
    for i,j in zip(range(4), range(4)):
        bpfilter[i, -(j+1)] = 1.

    C = np.sum((norm_mean_mut2*bpfilter).reshape(seqlen,seqlen,dims*dims), axis=2)
    #C = C - np.mean(C)
    #C = C/np.max(C)
    ugSS, numbp, numug, bpugSQ = htf.pkhp_SS()
    if datatype == '6' and not TRANSFER:
      ugSS = ugSS[1] #only extract the non-nested base pairs
      numbp = 3
    #get base pairing scores
    totscore = bd.bp_totscore(ugSS, C, numug)
    ppv = bd.bp_ppv(C, ugSS, numbp, numug)

    color = 'Oranges'

    plt.figure(figsize=(8,6))
    sb.heatmap(C, vmin=None, cmap=color , linewidth=0.00)
    plt.title('Base Pair scores: %s %s '%(exp, modelsavename))

    if JUSTPKHP:
      som_file = modelsavename + 'SoM_bpfilter' + '_justkhp' + '.png'
    if TRANSFER:
      som_file = modelsavename + 'SoM_bpfilter' + '_posttransfer' + '.png'
    else:
      som_file = modelsavename + 'SoM_bpfilter' + '_pretransfer' +'.png'

    som_file = os.path.join(img_folder, som_file)
    plt.savefig(som_file)
    plt.close()

    if WRITE:
      tran = 'notransfer'
      if TRANSFER:
          tran = 'transfer'
      numpos = len(X_train)//2
      metricsline = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(modelarch, datatype, trialnum, numepochs, tran,
                        numpos, loss, mean_vals[0], mean_vals[1], mean_vals[2], totscore, ppv)
      fd = open('test_metrics.csv', 'a')
      fd.write(metricsline+'\n')
      fd.close()

#---------------------------------------------------------------------------------------------
