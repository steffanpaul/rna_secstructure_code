import os
from subprocess import call

call(['python', 'mlp_varfam.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])
call(['python', 'resbind_varfam.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])