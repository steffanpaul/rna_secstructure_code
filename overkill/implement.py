import os
from subprocess import call

hiddenlist = ['2', '44', '196', '1024']

for numhidden in hiddenlist:
    call(['python', 'mlp_overkill.py', numhidden, '--train', '--test', '--somvis', '--somcalc'])

for numhidden in hiddenlist:
    call(['python', 'mlp_overkill.py', numhidden, '--test'])
