import os
from subprocess import call

trials = [['mlp_pkhp.py', '6'],
            ['resbind_pkhp.py', '6'],
            ['rnn_pkhp.py', '5']]
 def T1(file, data):
     call(['python', file, data, '1', '--justpkhp', '--test', '--somvis'])

 def T13(file, data, e=True):
     if e:
         call(['python', file, data, '1.3', '--some', '20', '--setepochs', '1000', '--justpkhp', '--test', '--somvis'])
