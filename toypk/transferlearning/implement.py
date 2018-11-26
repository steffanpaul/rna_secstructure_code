import os
from subprocess import call

#The first and second number after the filename are the datatype and the trial number
for file in ['rnn_pkhp.py']:
    for data in ['6']:
        if 'mlp' in file:
            '''
            #pretransfer - on hp
            #call(['python', file, data, '0', '--pretransfer', '--train', '--test', '--somcalc', '--somvis'])

            #without transferring - on pkhp
            #call(['python', file, data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '1.3', '--some', '20', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.4', '--some', '50', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.5', '--some', '100', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.6', '--some', '200', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])

            #with transfer
            #call(['python', file, data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '3', '--some', '20', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '4', '--some', '50', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '5', '--some', '100', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '6', '--some', '200', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

            #call(['python', file, data, '7', '--some', '500', '--transfer', '--setepochs', '2000',  '--train', '--test', '--somcalc', '--somvis'])

            #with transfer and extra training
            #call(['python', file, data, '5', '--some', '100', '--transfer', '--setepochs', '1000', '--test', '--somvis'])
            #call(['python', file, data, '6', '--some', '200', '--transfer', '--setepochs', '1000', '--train', '--test', '--somcalc', '--somvis'])


        if 'resbind' in file:

            #pretransfer - on hp
            #call(['python', file, data, '0', '--pretransfer', '--train', '--test', '--somcalc', '--somvis'])

            #without transferring - on pkhp
            #call(['python', file, data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.3', '--some', '20', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.4', '--some', '50', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.5', '--some', '100', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.6', '--some', '200', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])

            #without transfer - on pkhp - with extra training
            #call(['python', file, data, '1.4', '--some', '50', '--setepochs', '2000', '--justpkhp', '--test'])#'--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.5', '--some', '100', '--setepochs', '2000', '--justpkhp',  '--test'])#'--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '1.6', '--some', '200', '--setepochs', '1000', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '1.7', '--some', '500', '--setepochs', '1000', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])


            #with transfer
            #call(['python', file, data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '3', '--some', '20', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '4', '--some', '50', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '5', '--some', '100', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '6', '--some', '200', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

            #with transfer and extra training
            #call(['python', file, data, '2', '--transfer', '--setepochs', '1000', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '4', '--some', '50', '--setepochs', '2000', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer',  '--test'])#'--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '6', '--some', '200', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            call(['python', file, data, '7', '--some', '500', '--setepochs', '1000', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            '''


        if 'rnn' in file:
            #pretransfer - on hp
            #call(['python', file, data, '0', '--pretransfer', '--train', '--somcalc', '--somvis'])

            #without transferring - on pkhp
            #call(['python', file, data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '1.3', '--some', '20', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])

            #without transferring - on pkhp - extra training
            call(['python', file, data, '1.3', '--some', '20', '--setepochs', '1000', '--justpkhp', '--train', '--somcalc', '--somvis'])
            call(['python', file, data, '1.4', '--some', '50', '--setepochs', '1000', '--justpkhp', '--train', '--somcalc', '--somvis'])
            call(['python', file, data, '1.5', '--some', '100', '--setepochs', '1000', '--justpkhp', '--train', '--somcalc', '--somvis'])

            #with transfer
            #call(['python', file, data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
            #call(['python', file, data, '3', '--some', '20', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

            #with transfer and extra training
            call(['python', file, data, '3', '--some', '20', '--setepochs', '1000', '--transfer', '--train', '--somcalc', '--somvis'])
            call(['python', file, data, '4', '--some', '50', '--setepochs', '1000', '--transfer', '--train', '--somcalc', '--somvis'])
            call(['python', file, data, '5', '--some', '100', '--setepochs', '1000', '--transfer', '--train', '--somcalc', '--somvis'])
