import os
from subprocess import call

call(['python', 'mlp_trainsize.py', '--train', '--test', '--somcalc', '--somvis'])
call(['python', 'resbind_trainsize.py', '--train', '--test', '--somcalc', '--somvis'])