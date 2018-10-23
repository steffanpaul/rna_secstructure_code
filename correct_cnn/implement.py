import os
from subprocess import call


call(['python', 'resbind_fam.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])
call(['python', 'resbind_varfam.py', '--train', '--test', '--fom', '--somcalc', '--somvis'])