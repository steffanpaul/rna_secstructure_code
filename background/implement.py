import os
from subprocess import call


#call(['python', 'background_factory.py'])
call(['python', 'mlp_background.py', '--train', '--test', '--somcalc', '--somvis'])