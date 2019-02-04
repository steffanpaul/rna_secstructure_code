import os, sys
from subprocess import call

families =     ['RF00059', 'RF00169', 'RF00174', 'RF00504']#, 'RF01960'] 'RF00002', 'RF00005', 'RF00010', 'RF00017', 'RF00023',

trainsizes = ['4000', '2000', '1000', '500', '250', '100']
#stopped at RF00059 after SoM on ts2000. Continue on training ts1000

for fam in families:
    if fam == 'RF00059':
        for ts in ['1000', '500', '250', '100']:
            call(['python', 'mlp_marks_ts.py', fam, ts, '512', '--train', '--test', '--somcalc', '--write'])
    else:
        for ts in trainsizes:
            call(['python', 'mlp_marks_ts.py', fam, ts, '512', '--train', '--test', '--somcalc', '--write'])
