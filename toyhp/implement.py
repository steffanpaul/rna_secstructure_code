import os
from subprocess import call
#for ii in range(3):
    #call(['python', 'rnn_sizes_toyhp_saliency.py', '--write'])
    #call(['python', 'resbind_sizes_toyhp_saliency.py','--test', '--write'])
    #call(['python', 'mlp_sizes_toyhp_saliency.py','--test', '--write'])

#call(['python', 'resbind_varsizes_toyhp_saliency.py', 
#      '--somcalc', '--somvis'])

#call(['python', 'mlp_varsizes_toyhp_train.py'])
#call(['python', 'mlp_varsizes_toyhp_saliency.py', 
#     '--test', '--write', '--fom', '--somcalc', '--somvis'])

call(['python', 'rnn_varsizes_toyhp_train.py'])
call(['python', 'rnn_varsizes_toyhp_saliency.py', 
     '--write', '--somcalc', '--somvis'])


call(['python', 'rnn_sizes_toyhp_train.py'])
call(['python', 'rnn_sizes_toyhp_saliency.py', 
     '--write', '--somcalc', '--somvis'])

call(['python', 'mlp_varfam.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])
call(['python', 'resbind_varfam.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])

