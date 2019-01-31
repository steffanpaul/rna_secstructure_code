import os
from subprocess import call

families = ['RF00002', 'RF00005', 'RF00010', 'RF00017', 'RF00023', 'RF00050',
            'RF00059', 'RF00162', 'RF00167', 'RF00169', 'RF00174', 'RF00234',
            'RF00380', 'RF00504', 'RF01734', 'RF01786', 'RF01831', 'RF01852', 'RF01960', 'RF02001']

rnn_tests_full = ['RF00005']
rnn_tests_som = ['RF00059', 'RF00174', 'RF00005']

for fam in rnn_tests_som:
    call(['python', 'rnn_marks.py', fam, '--somcalc', '--somvis', '--write'])

#for fam in rnn_tests_full:
#    call(['python', 'rnn_marks.py', fam, '--train', '--somcalc', '--somvis', '--write'])
    #call(['python', 'test.py', fam])
