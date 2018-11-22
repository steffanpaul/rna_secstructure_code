N = numdata
posidx = np.random.permutation(np.arange(N//2))
negidx = np.random.permutation(np.arange(N//2, N))
split_1 = int((N//2)*(1-valid_frac-test_frac))
split_2 = int((N//2)*(1-test_frac))
#shuffle = np.random.permutation(N)

trainidx = np.random.permutation(np.concatenate([posidx[:split_1], negidx[:split_1]]))
valididx = np.random.permutation(np.concatenate([posidx[split_1:split_2], negidx[split_1:split_2]]))
testidx = np.random.permutation(np.concatenate([posidx[split_2:], negidx[split_2:]]))

#set up dictionaries
train = {'inputs': X_data[trainidx],
         'targets': Y_data[trainidx]}
valid = {'inputs': X_data[valididx],
         'targets': Y_data[valididx]}
test = {'inputs': X_data[testidx],
         'targets': Y_data[testidx]}





N = numdata
split_1 = int(N*(1-valid_frac-test_frac))
split_2 = int(N*(1-test_frac))
shuffle = np.random.permutation(N)

#set up dictionaries
train = {'inputs': X_data[shuffle[:split_1]],
         'targets': Y_data[shuffle[:split_1]]}
valid = {'inputs': X_data[shuffle[split_1:split_2]],
         'targets': Y_data[shuffle[split_1:split_2]]}
test = {'inputs': X_data[shuffle[split_2:]],
         'targets': Y_data[shuffle[split_2:]]}
print (len(train['inputs']))
