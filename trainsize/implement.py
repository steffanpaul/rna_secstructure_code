import os
from subprocess import call

call(['python', 'mlp_trainsize.py', '--test', '--somvis', '--write'])
call(['python', 'resbind_trainsize.py', '--test', '--somvis', '--write'])
