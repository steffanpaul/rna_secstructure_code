import os
from subprocess import call

#The first and second number after the filename are the datatype and the trial number

file = 'mlp_pkhp.py'

for data, pretrainepochs in [['6.01', '1'], ['6.02', '2'], ['6.03', '3']]:
    #pretransfer - on hp
    call(['python', file, data, '0', '--pretransfer', '--setepochs', pretrainepochs, '--train', '--test', '--somcalc', '--somvis'])

    #with transfer
    call(['python', file, data, '7', '--some', '500', '--setepochs', '2000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])


file = 'resbind_pkhp.py'

for data, pretrainepochs in [['6.01', '1'], ['6.03', '3'], ['6.05', '5']]:
    #pretransfer - on hp
    call(['python', file, data, '0', '--pretransfer', '--setepochs', pretrainepochs, '--train', '--test', '--somcalc', '--somvis'])

    #with transfer
    call(['python', file, data, '2', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '4', '--some', '50', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
