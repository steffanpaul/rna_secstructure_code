import os
from subprocess import call

#The first and second number after the filename are the datatype and the trial number
'''
file = 'mlp_pkhp.py'

for data, pretrainepochs in [['6.01', '1'], ['6.02', '2'], ['6.03', '3']]:
    #pretransfer - on hp
    call(['python', file, data, '0', '--pretransfer', '--setepochs', pretrainepochs, '--train', '--test', '--somcalc', '--somvis'])

    #with transfer
    call(['python', file, data, '7', '--some', '500', '--setepochs', '2000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])


file = 'resbind_pkhp.py'

for data, pretrainepochs in [['6.20', '20']]:#['6.01', '1'], ['6.03', '3'], ['6.05', '5']]:
    #pretransfer - on hp
    #call(['python', file, data, '0', '--pretransfer', '--setepochs', pretrainepochs, '--train', '--test', '--somcalc', '--somvis'])

    #with transfer
    #call(['python', file, data, '2', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '4', '--some', '50', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
'''
file = 'rnn_pkhp.py'

for data, pretrainepochs in [['5.100', '100']]:#, ['5.01', '1'], ['6.03', '3'], ['6.05', '5']]:
    #pretransfer - on hp
    #call(['python', file, data, '0', '--pretransfer', '--setepochs', pretrainepochs, '--train', '--test', '--somcalc', '--somvis'])

    #call(['python', file, data, '1', '--setepochs', '500', '--justpkhp', '--train', '--test', '--somcalc', '--somvis', '--write'])
    #call(['python', file, data, '1.3', '--some', '20', '--setepochs', '500', '--justpkhp', '--train', '--test', '--somcalc', '--somvis', '--write'])
    #call(['python', file, data, '1.4', '--some', '20', '--setepochs', '500', '--justpkhp', '--train', '--test', '--somcalc', '--somvis', '--write'])


    #with transfer
    SOMCAL SOM VIS call(['python', file, data, '2', '--setepochs', '500', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '3', '--some', '20', '--setepochs', '500', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '4', '--some', '50', '--setepochs', '500', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
    call(['python', file, data, '5', '--some', '100', '--setepochs', '500', '--transfer', '--train', '--test', '--somcalc', '--somvis', '--write'])
