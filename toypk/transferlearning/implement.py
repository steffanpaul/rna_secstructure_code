import os
from subprocess import call


call(['python', 'mlp_pkhp.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])
call(['python', 'mlp_pkhp.py', '--transfer', '--train', '--test', '--fom', '--somcalc', '--somvis'])

