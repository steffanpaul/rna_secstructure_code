import os
from subprocess import call

hiddenlist = ['88', '128', '512']

run = False
test = False
vis = True

if run:
    for numhidden in hiddenlist:
        call(['python', 'mlp_riboswitch_2layer.py', numhidden, '--mode', 'full','--somvis', '--somcalc', '--write'])
        call(['python', 'mlp_riboswitch_2layer.py', numhidden, '--mode', 'sim2k','--somvis', '--somcalc', '--write'])
        call(['python', 'mlp_riboswitch_2layer.py', numhidden, '--mode', 'sim2k_cut','--somvis', '--somcalc', '--write'])

if test:
    for numhidden in hiddenlist:
        call(['python', 'mlp_riboswitch_2layer.py', numhidden, '--mode', 'full', '--test', '--write'])
        call(['python', 'mlp_riboswitch_2layer.py', numhidden, '--mode', 'sim2k', '--test', '--write'])
        call(['python', 'mlp_riboswitch_2layer.py', numhidden, '--mode', 'sim2k_cut','--test', '--write'])

if vis:
    for numhidden in hiddenlist:
        call(['python', 'mlp_riboswitch.py', numhidden, '--mode', 'full', '--somvis', '--apc'])
        #call(['python', 'mlp_riboswitch.py', numhidden, '--mode', 'sim2k','--somvis', '--apc'])
        #call(['python', 'mlp_riboswitch.py', numhidden, '--mode', 'sim2k_cut','--somvis', '--apc'])
