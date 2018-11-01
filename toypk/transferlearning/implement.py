import os
from subprocess import call

#The first and second number after the filename are the datatype and the trial number

for data in ['4']:
	call(['python', 'mlp_pkhp.py', data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])
	
	call(['python', 'mlp_pkhp.py', data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

	call(['python', 'mlp_pkhp.py', data, '3', '50', '--some', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

	call(['python', 'mlp_pkhp.py', data, '4', '100', '--some', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

	call(['python', 'mlp_pkhp.py', data, '5', '200', '--some', '--transfer', '--train', '--test', '--somcalc', '--somvis'])


	'''
	#train on just pkhp
	call(['python', 'mlp_pkhp.py', data, '1', '--justpkhp', '--train', '--test', '--somcalc', '--somvis'])

	#train on all hp and then transfer to all pkhp
	call(['python', 'mlp_pkhp.py', data, '2', '--train', '--test', '--somcalc', '--somvis'])
	call(['python', 'mlp_pkhp.py', data, '2', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

	#train on some pkhp
	call(['python', 'mlp_pkhp.py', data, '3', '--some', '--train', '--test', '--somcalc', '--somvis'])

	#train on all hp and then transfer to some pkhp
	call(['python', 'mlp_pkhp.py', data, '4', '--train', '--test', '--somcalc', '--somvis'])
	call(['python', 'mlp_pkhp.py', data, '4', '--some', '--transfer', '--train', '--test', '--somcalc', '--somvis'])

	#train on all hp and then transfer to even less pkhp (1000pos and 1000neg)
	call(['python', 'mlp_pkhp.py', data, '6', '--train', '--test'])#, '--somcalc', '--somvis'])
	call(['python', 'mlp_pkhp.py', data, '6', '--some', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
	
	#train on all hp and then transfer to even less pkhp (1000pos and 1000neg)
	call(['python', 'mlp_pkhp.py', data, '7', '--train', '--test'])#, '--somcalc', '--somvis'])
	call(['python', 'mlp_pkhp.py', data, '7', '--some', '--transfer', '--train', '--test', '--somcalc', '--somvis'])
'''