import os
from subprocess import call

from subprocess import call
call(['python', 'background_gap_factory.py'])
call(['python', 'mlp_background_gap.py', '--train', '--test', '--somcalc', '--somvis'])
call(['python', 'resbind_background_gap.py', '--train', '--test', '--somcalc', '--somvis'])