import os
from subprocess import call


call(['python', 'resbind_pkhp.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])
call(['python', 'resbind_pkhp.py', '--transfer', '--train', '--test', '--fom', '--somcalc', '--somvis'])

