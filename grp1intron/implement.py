import os
from subprocess import call

hiddenlist = ['512', '1024','44', '128']

for numhidden in hiddenlist:
    call(['python', 'mlp_grp1intron.py', numhidden, '--somvis'])
    #call(['python', 'mlp_grp1intron.py', numhidden, '--somvis', '--apc'])
#for numhidden in hiddenlist:
#    call(['python', 'mlp_grp1intron.py', numhidden, '--test'])
