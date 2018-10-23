import os
from subprocess import call

#The first and second number after the filename are the datatype and the trial number

for data in ['1','2']:
	#train on just pkhp
	call(['python', 'mlp_pkhp.py', data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])

	#train on all hp and then transfer to all pkhp
	call(['python', 'mlp_pkhp.py', data, '2', '--train', '--test', '--somcalc', '--somvis'])
	call(['python', 'mlp_pkhp.py', data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

	#train on some pkhp
	call(['python', 'mlp_pkhp.py', data, '3', '--some', '--train', '--test', '--somcalc', '--somvis'])

	#train on all hp and then transfer to some pkhp
	call(['python', 'mlp_pkhp.py', data, '4', '--train', '--test', '--somcalc', '--somvis'])
	call(['python', 'mlp_pkhp.py', data, '4', '--some', '--transfser', '--train', '--test', '--somcalc', '--somvis'])
