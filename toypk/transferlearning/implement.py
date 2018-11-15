import os
from subprocess import call

#The first and second number after the filename are the datatype and the trial number

for data in ['5', '6']:
    #pretransfer - on hp
    call(['python', 'mlp_pkhp.py', data, '0', '--pretransfer', '--train', '--test', '--somcalc', '--somvis'])

    #without transferring - on pkhp
    call(['python', 'mlp_pkhp.py', data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
    call(['python', 'mlp_pkhp.py', data, '1.4', '--some', '100', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
    call(['python', 'mlp_pkhp.py', data, '1.5', '--some', '200', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])

    #with transfer
    call(['python', 'mlp_pkhp.py', data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
    call(['python', 'mlp_pkhp.py', data, '3', '--some', '50', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
    call(['python', 'mlp_pkhp.py', data, '4', '--some', '100', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
    call(['python', 'mlp_pkhp.py', data, '5', '--some', '200', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

    #with transfer and extra training
    call(['python', 'mlp_pkhp.py', data, '4.1', '--some', '100', '--transfer', '--setepochs', '1000', '--train', '--test', '--somcalc', '--somvis'])
    call(['python', 'mlp_pkhp.py', data, '5.1', '--some', '200', '--transfer', '--setepochs', '1000', '--train', '--test', '--somcalc', '--somvis'])