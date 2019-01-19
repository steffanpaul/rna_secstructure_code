import os
from subprocess import call

hiddenlist = ['44', '128', '512', '1024']
families = ['chrB-a', 'C4', 'int-alpA', 'LOOT', 'manA', 'RAGATH-18', 'RAGATH-2-HDV', 'chrB-b']


for numhidden in hiddenlist:
    for fam in families:
        call(['python', 'mlp_zwein.py', fam, numhidden, '--train', '--test', '--somcalc', '--somvis', '--write'])
