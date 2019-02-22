import os,sys
from subprocess import call

call(['python', 'rnn_tpp.py', '--train', '--somcalc', '--somvis', '--write'])
#call(['python', 'mlp_tpp.py', '512', '--train', '--test', '--somcalc', '--somvis', '--write'])
