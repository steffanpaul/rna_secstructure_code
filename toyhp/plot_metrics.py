import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

filename = 'test_metrics.csv'

labels = []
CNN = []
MLP = []
RNN = []
for ii, line in enumerate(open(filename, 'r')):
	line = line.strip('\n')
	line = line.split(',')
	if ii == 0:
		for l in line:
			labels.append(l)
	else:
		if 'med' in line:
			if 'resbind' in line:
				CNN.append([float(l) for l in line[4:]])
			elif 'mlp' in line:
				MLP.append([float(l) for l in line[4:]])
			elif 'rnn' in line:
				RNN.append([float(l) for l in line[4:]])

CNNmeans = np.mean(CNN, axis=0)
MLPmeans = np.mean(MLP, axis=0)
RNNmeans = np.mean(RNN, axis=0)

CNNstds = np.std(CNN, axis=0)
MLPstds = np.std(MLP, axis=0)
RNNstds = np.std(RNN, axis=0)

N = len(CNNmeans)
width  = 0.15
ind = np.arange(N)

fig, ax = plt.subplots()

rects1 = ax.errorbar(ind, CNNmeans, color='r', yerr=CNNstds, fmt='-', capsize=5.)

rects2 = ax.errorbar(ind, MLPmeans, color='g', yerr=MLPstds, fmt='-', capsize=5.)

rects3 = ax.errorbar(ind, RNNmeans, color='b', yerr=RNNstds, fmt='-', capsize=5.)

# add some text for labels, title and axes ticks
ax.set_ylabel('Score')
ax.set_title('Performance metrics across models')
ax.set_xticks(ind)
ax.set_xticklabels(('Accurary', 'ROC-AUC', 'ROC-PR'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('CNN', 'MLP', 'RNN'))





import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

filename = 'test_metrics.csv'

labels = []
CNN = []
MLP = []
RNN = []
for ii, line in enumerate(open(filename, 'r')):
	line = line.strip('\n')
	line = line.split(',')
	if ii == 0:
		for l in line:
			labels.append(l)
	else:
		if 'med' in line:
			if 'resbind' in line:
				CNN.append([float(l) for l in [line[3]]])
			elif 'mlp' in line:
				MLP.append([float(l) for l in [line[3]]])
			elif 'rnn' in line:
				RNN.append([float(l) for l in [line[3]]])

CNNmeans = np.mean(CNN, axis=0)
MLPmeans = np.mean(MLP, axis=0)
RNNmeans = np.mean(RNN, axis=0)

CNNstds = np.std(CNN, axis=0)
MLPstds = np.std(MLP, axis=0)
RNNstds = np.std(RNN, axis=0)

N = len(CNNmeans)
width  = 0.15
ind = np.arange(N)

fig, ax = plt.subplots()

rects1 = ax.errorbar(ind, CNNmeans, color='r', yerr=CNNstds, fmt='o')

rects2 = ax.errorbar(ind, MLPmeans, color='g', yerr=MLPstds, fmt='o')

rects3 = ax.errorbar(ind, RNNmeans, color='b', yerr=RNNstds, fmt='o')

# add some text for labels, title and axes ticks
ax.set_ylabel('Score')
ax.set_title('Performance metrics across models')
ax.set_xticks(ind)
ax.set_xticklabels(('Accurary', 'ROC-AUC', 'ROC-PR'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('CNN', 'MLP', 'RNN'))

