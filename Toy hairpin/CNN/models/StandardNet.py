
def model(input_shape, dropout=True, l2=True, batch_norm=True):

    layer1 = {  'layer': 'input',
                'input_shape': input_shape
             }
    layer2 = {  'layer': 'conv1d',
                'num_filters': 24,
                'filter_size': 19,
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3, 
                }
    if batch_norm:
        layer2['norm'] = 'batch'
    layer3 = {  'layer': 'conv1d',
                'num_filters': 48,
                'filter_size': 8,
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3,  
                }
    if dropout:
        layer3['dropout'] = 0.2
    if batch_norm:
        layer3['norm'] = 'batch'
    layer4 = {  'layer': 'conv1d',
                'num_filters': 96,
                'filter_size': 8,
                'padding': 'VALID',
                'activation': 'relu',
                'max_pool': 3, 
                }
    if dropout:
        layer4['dropout'] = 0.2
    if batch_norm:
        layer4['norm'] = 'batch'
    layer5 = {  'layer': 'dense',
                'num_units': 128,
                'activation': 'relu',
                }
    if dropout:
        layer4['dropout'] = 0.5
    if batch_norm:
        layer5['norm'] = 'batch'
    layer6 = {  'layer': 'dense',
                'num_units': 128,
                'activation': 'relu',
                }
    if dropout:
        layer5['dropout'] = 0.5
    if batch_norm:
        layer6['norm'] = 'batch'
    layer7 = {  'layer': 'dense',
                'num_units': 1,
                'activation': 'sigmoid',
                }

    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.0003,
                  }
    if l2:
        optimization['l2'] = 1e-6

    return model_layers, optimization
