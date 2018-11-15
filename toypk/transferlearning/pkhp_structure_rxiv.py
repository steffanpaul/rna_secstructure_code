#


'''
First pkhp made. The structure used for d1-4. Contains two non-nested stems and one nested stem 
separate from the others
'''
def build_pkhp(stemsize=6, loopsize=5, numstems=3, numloops=7):
    loops = [randomsequence(1,loopsize)[0] for l in range(numloops)]
    
    stem1s = [] #The first binding region of each stem
    stem2s = [] #The second binding region of each stem
    for s in range(numstems):
        stem1, stem2 = stemregion(1, stemsize)
        stem1s.append(stem1[0])
        stem2s.append(stem2[0])
    #Add binding regions in the order of their place in the sequence
    stems = np.vstack([stem1s[0], stem1s[1], stem2s[0], stem1s[2], stem2s[2], stem2s[1]])

    #assemble
    pk_idx = []
    for ii in range(numloops):
        pk_idx=pk_idx + loops[ii]
        if ii <= len(stems)-1:
            pk_idx=pk_idx+list(stems[ii])
    return(pk_idx)


'''
Second pkhp made. More simple structure as suggested by Peter. Consists of two 
nested hairpins, with a non-nested binding region in their inner loops.
This will have to take in different sizes of stemsize and loopsize 
as everything is not equal.
'''
stemsizes = [5,5,3]
loopsizes = [5,2,2,10,2,2,5]
def build_pkhp(stemsizes=stemsizes, loopsizes=loopsizes, numstems=3, numloops=7):
    loops = [randomsequence(1,ls)[0] for ls in loopsizes]
    
    stem1s = [] #The first binding region of each stem
    stem2s = [] #The second binding region of each stem
    for sts in stemsizes:
        stem1, stem2 = stemregion(1, sts)
        stem1s.append(stem1[0])
        stem2s.append(stem2[0])
    #Add binding regions in the order of their place in the sequence
    stems = np.vstack([stem1s[0], stem1s[2], stem2s[0], stem1s[1], stem2s[2], stem2s[1]])

    #assemble
    pk_idx = []
    for ii in range(numloops):
        pk_idx=pk_idx + loops[ii]
        if ii <= len(stems)-1:
            pk_idx=pk_idx+list(stems[ii])
    return(pk_idx)