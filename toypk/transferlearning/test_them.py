import os
from subprocess import call

for iii in range(9):
    #The first and second number after the filename are the datatype and the trial number
    file = 'mlp_pkhp.py'
    data = '6'
    #without transferring - on pkhp
    call(['python', file, data, '1', '--justpkhp', '--test', '--somvis'])
    call(['python', file, data, '1.3', '--some', '20', '--justpkhp', '--test', '--somvis'])
    call(['python', file, data, '1.4', '--some', '50', '--justpkhp', '--test', '--somvis'])
    call(['python', file, data, '1.5', '--some', '100', '--setepochs', '1000', '--justpkhp', '--test',  '--somvis'])
    call(['python', file, data, '1.6', '--some', '200', '--setepochs', '1000', '--justpkhp', '--test',  '--somvis'])
    call(['python', file, data, '1.7', '--some', '500', '--setepochs', '2000', '--justpkhp', '--test',  '--somvis'])

    #with transfer
    call(['python', file, data, '2', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '3', '--some', '20', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '4', '--some', '50', '--transfer', '--test', '--somvis'])

    #with extra training
    call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '6', '--some', '200', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '7', '--some', '500', '--setepochs', '2000', '--transfer', '--test', '--somvis'])


    file = 'resbind_pkhp.py'
    data = '6'
    #without transferring - on pkhp
    call(['python', file, data, '1', '--justpkhp', '--test', '--somvis'])
    call(['python', file, data, '1.3', '--some', '20', '--setepochs', '1000', '--justpkhp', '--test',  '--somvis'])
    call(['python', file, data, '1.4', '--some', '50', '--setepochs', '2000', '--justpkhp', '--test',  '--somvis'])
    call(['python', file, data, '1.5', '--some', '100', '--setepochs', '2000', '--justpkhp', '--test',  '--somvis'])
    call(['python', file, data, '1.6', '--some', '200', '--setepochs', '1000', '--justpkhp', '--test',  '--somvis'])
    call(['python', file, data, '1.7', '--some', '500', '--setepochs', '1000', '--justpkhp', '--test',  '--somvis'])

    #with extra training
    call(['python', file, data, '2', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '4', '--some', '50', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '6', '--some', '200', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '7', '--some', '500', '--setepochs', '1000', '--transfer', '--test', '--somvis'])

'''
for file, data in ['rnn_pkhp.py', '5']:
    #without transferring - on pkhp
    call(['python', file, data, '1', '--justpkhp', '--test', '--somvis'])
    call(['python', file, data, '1.3', '--some', '20', '--setepochs', '1000', '--justpkhp', '--test',  '--somvis'])

    #with transfer
    call(['python', file, data, '2', '--transfer', '--test', '--somvis'])

    #with extra training
    call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
    call(['python', file, data, '4', '--some', '50', '--setepochs', '1000', '--transfer', '--test', '--somvis'])
'''
