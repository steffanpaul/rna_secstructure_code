import os
from subprocess import call


call(['python', 'background_factory.py'])
call(['python', 'resbind_background.py', '--train', '--test', '--somcalc', '--somvis'])