import os
from subprocess import call

file = 'rnn_pkhp.py'
data = '5'

#64 hidden layer
#call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

#32 hidden layer
call(['python', file, data, '0', '--sethidden', '32', '--pretransfer', '--train', '--test', '--somcalc', '--somvis'])
call(['python', file, data, '1', '--setepochs', '1000', '--sethidden', '32', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
call(['python', file, data, '2', '--setepochs', '1000', '--sethidden', '32', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

#128 hidden layer
call(['python', file, data, '0', '--sethidden', '128', '--pretransfer', '--train', '--test', '--somcalc', '--somvis'])
call(['python', file, data, '1', '--setepochs', '1000', '--sethidden', '128', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
call(['python', file, data, '2', '--setepochs', '1000', '--sethidden', '128', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
