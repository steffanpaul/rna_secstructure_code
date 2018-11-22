trainportion_list = [0.7, 0.5, 0.3, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.001]

t = 'trna'
for trainportion in trainportion_list:
    if trainportion >= 0.1:
        tp = '_tp%.0f'%(trainportion*100)
    elif trainportion >= 0.01:
        tp = '_tp0%.0f'%(trainportion*100)
    elif trainportion >= 0.001:
        tp = '_tp00%.0f'%(trainportion*1000)

    trial = t + tp
    print (trial)
