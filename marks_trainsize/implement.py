import os, sys
from subprocess import call

families =     ['RF00002', 'RF00005', 'RF00010', 'RF00017', 'RF00023',
                'RF00059', 'RF00169', 'RF00174', 'RF00504']#, 'RF01960']

trainsizes = ['1250', '800', '600', '300']#['4000', '2000', '1000', '500', '250', '100']

for fam in families:
    for ts in trainsizes:
        call(['python', 'mlp_marks_ts.py', fam, ts, '512', '--train', '--test', '--somcalc', '--write'])
