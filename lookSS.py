import os, sys

filename = sys.argv[1]

with open(filename, 'r') as fd:
    for line in fd:
        line = line.strip('\n')
        if '#=GC SS_cons' in line:
            print line
    fd.close()
