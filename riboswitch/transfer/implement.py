import os
from subprocess import call

hiddenlist = ['128', '512']

for numhidden in hiddenlist:
    #call(['python', 'mlp_riboswitch_full.py', numhidden, '--train', '--test', '--somvis', '--somcalc'])
    call(['python', 'mlp_riboswitch_sim2k.py', numhidden, '--train', '--test', '--somvis', '--somcalc'])
    #call(['python', 'mlp_riboswitch_sim2k_cut.py', numhidden, '--train', '--test', '--somvis', '--somcalc'])

for numhidden in hiddenlist:
    call(['python', 'mlp_riboswitch_full.py', numhidden, '--test'])
    call(['python', 'mlp_riboswitch_sim2k.py', numhidden, '--test'])
    call(['python', 'mlp_riboswitch_sim2k_cut.py', numhidden, '--test'])
