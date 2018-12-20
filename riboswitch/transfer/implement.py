import os
from subprocess import call

for numhidden in ['22', '30', '88', '128']:
    call(['python', 'mlp_riboswitch.py', numhidden, '--train', '--test', '--somvis', '--somcalc'])

for numhidden in ['22', '30', '88', '128']:
    call(['python', 'mlp_riboswitch.py', numhidden, '--test'])
