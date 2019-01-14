import os
from subprocess import call

hiddenlist = ['44', '128', '512', '1024']

for numhidden in hiddenlist:
    call(['python', 'mlp_grp1intron.py', numhidden, '--train', '--test', '--somvis', '--somcalc'])

for numhidden in hiddenlist:
    call(['python', 'mlp_grp1intron.py', numhidden, '--test'])
