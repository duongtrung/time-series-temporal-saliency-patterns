


def create_inout_sequences(input_data, input_window, output_window):
    import numpy as np
    import torch
    inout_seq = []
    L = len(input_data)
    
    for i in range(L - input_window):
        train_seq = np.append(input_data[i:i + input_window][:-output_window], output_window * [0])
        train_label = input_data[i:i + input_window]
        inout_seq.append((train_seq, train_label))
    
    return torch.FloatTensor(inout_seq)


def create_inout_sequences_mul(input_data, input_label, input_window, output_window, number_of_cols):
    import numpy as np
    import torch
    inout_seq = []
    L = len(input_label)
    
    for i in range(L - input_window):
        input_data_loop = input_data[i*number_of_cols:i*number_of_cols + input_window*number_of_cols]
        train_seq = np.append(input_data_loop[:-output_window*number_of_cols], output_window * number_of_cols * [0])
        train_label = np.repeat(input_label[i:i + input_window],number_of_cols)
        for k in range(len(train_label)):
            if k !=0 and k % number_of_cols != 0:
                train_label[k-1] = 0
        inout_seq.append((train_seq, train_label))
    
    return torch.FloatTensor(inout_seq)


def get_ETTh1_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from Informer paper
    train_data = series[0:8760]
    valid_data = series[8760:11712]
    test_data = series[11712:14591]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ETTh1_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()

    train_split = 8760
    valid_split = 11712
    test_split = 14591
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ETTh1_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/ETT-small/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    # 7:1:2 as in Autoformer
    train_split = 12194
    valid_split = 13936
    test_split = 17420
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ETTm1_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/ETT-small/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    # 7:1:2 as in Autoformer
    train_split = 48776
    valid_split = 55744
    test_split = 69680
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)
    

def get_ETTh2_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/ETT-small/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    # 7:1:2 as in Autoformer
    train_split = 12194
    valid_split = 13936
    test_split = 17420
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)
    


def get_ETTh2_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from Informer paper
    train_data = series[0:8760]
    valid_data = series[8760:11712]
    test_data = series[11712:14591]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ETTh2_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()

    train_split = 8760
    valid_split = 11712
    test_split = 14591
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    

    

def get_ETTm1_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from Informer paper
    train_data = series[0:35040]
    valid_data = series[35040:46848]
    test_data = series[46848:58368]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ETTm2_AF_uni(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
       
    series = read_csv('data/Autoformer/ETT-small/' + dataset, header=0)
    series = series[target]
    from sklearn.preprocessing import StandardScaler, RobustScaler
    scaler = RobustScaler()

    # 6:2:2 as in Autoformer paper
    train_data = series[0:41808]
    valid_data = series[41808:55744]
    test_data = series[55744:69680]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)

    
'''
def get_ETTm2_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from N-HiTS
    train_data = series[0:41808]
    valid_data = series[41808:55744]
    test_data = series[55744:69680]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)
'''

def get_ETTm1_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()

    train_split = 35040
    valid_split = 46848
    test_split = 58368
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    
    

def get_Ili_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    #series = read_csv('data/Autoformer/illness/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = read_csv('data/Autoformer/illness/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    train_split = 676   # 7:1:2 as in Autoformer paper
    valid_split = 773
    test_split = 966
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    #print("test_data")
    #print(test_data)
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    #print("test_label")
    #print(test_label)
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    


def get_Exchange_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/exchange_rate/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    train_split = 5311   # 7:1:2 as in Autoformer paper
    valid_split = 6070
    test_split = 7588
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    #print("test_data")
    #print(test_data)
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    #print("test_label")
    #print(test_label)
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    
    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)   


def get_Electricity_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/electricity/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    train_split = 18413   # 7:1:2 as in Autoformer paper
    valid_split = 21043
    test_split = 26304
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    #print("test_data")
    #print(test_data)
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    #print("test_label")
    #print(test_label)
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)   
    

def get_Weather_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/weather/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    train_split = 36887   # 7:1:2 as in Autoformer paper
    valid_split = 42157
    test_split = 52696
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    #print("test_data")
    #print(test_data)
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    #print("test_label")
    #print(test_label)
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)       
    

def get_ETTm2_AF_mul(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/Autoformer/ETT-small/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    train_split = 41808   # 6:2:2 as in Autoformer paper
    valid_split = 55744
    test_split = 69680
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    



def get_ECL_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from Informer paper
    train_data = series[10200:21168]
    valid_data = series[21168:23376]
    test_data = series[23376:26304]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ECL_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()

    train_data = series[10200:21168][:]
    valid_data = series[21168:23376][:]
    test_data = series[23376:26304][:]
    
    train_label = labels[10200:21168]
    valid_label = labels[21168:23376]
    test_label = labels[23376:26304]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    


def get_ECL_n_hits_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/n-hits/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()
    
    # 7:1:2 as in N-HiTS paper
    train_data = series[0:18413][:]
    valid_data = series[18413:21043][:]
    test_data = series[21043:26304][:]
    
    train_label = labels[0:18413]
    valid_label = labels[18413:21043]
    test_label = labels[21043:26304]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    
    
    
def get_ETTm2_n_hits_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    
    series = read_csv('data/n-hits/' + dataset, header=0)
    series = series[series.columns.drop('date')]
    labels = series[target]
    from sklearn.preprocessing import StandardScaler, RobustScaler
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()
    #scaler_data = RobustScaler()
    #scaler_label = RobustScaler()
    
    # 6:2:2 as in N-HiTS paper
    train_data = series[0:34560][:]
    valid_data = series[34560:46080][:]
    test_data = series[21043:57600][:]
    
    train_label = labels[0:34560]
    valid_label = labels[34560:46080]
    test_label = labels[46080:57600]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    
     
    
    
def get_Weather_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from Informer paper
    train_data = series[0:20424]
    valid_data = series[20424:27720]
    test_data = series[27720:35064]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)
    
    
def get_Weather_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler, RobustScaler
    scaler_data = StandardScaler()
    scaler_label = StandardScaler()
    
    train_split = 20424
    valid_split = 27720
    test_split = 35064
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)    


def get_Exchange_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/n-hits/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler, RobustScaler
    #scaler_data = StandardScaler()
    #scaler_label = StandardScaler()
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()
    

    train_data = series[0:5311][:]
    valid_data = series[5311:6070][:]
    test_data = series[6070:7588][:]
    
    train_label = labels[0:5311]
    valid_label = labels[5311:6070]
    test_label = labels[6070:7588]
    

    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)   
    
    
def get_Exchange_data(dataset, target, input_window, output_window, device):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    series = read_csv('data/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = series[target]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    # The split are from N-HiTS 70-10-20
    train_data = series[0:5311]
    valid_data = series[5311:6070]
    test_data = series[6070:7588]
    
    train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences(valid_data, input_window, output_window)
    valid_sequence = valid_sequence[:-output_window]

    test_sequence = create_inout_sequences(test_data, input_window, output_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)


def get_ILI_mul_data(dataset, target, input_window, output_window, device, number_of_cols):
    from pandas import read_csv
    dataset = dataset
    target=target
    input_window = input_window
    output_window = output_window
    device = device
    number_of_cols = number_of_cols
    series = read_csv('data/n-hits/' + dataset, header=0, index_col=0, parse_dates=True, squeeze=True)
    labels = series[target]
    from sklearn.preprocessing import StandardScaler, RobustScaler
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()
    
    train_split = 676
    valid_split = 773
    test_split = 966
    
    train_data = series[:train_split][:]
    valid_data = series[train_split : valid_split][:]
    test_data = series[valid_split : test_split][:]
    
    train_label = labels[:train_split]
    valid_label = labels[train_split: valid_split]
    test_label = labels[valid_split: test_split]
    
    train_data = scaler_data.fit_transform(train_data.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_data = scaler_data.transform(valid_data.to_numpy().reshape(-1, 1)).reshape(-1)
    test_data = scaler_data.transform(test_data.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_label = scaler_label.fit_transform(train_label.to_numpy().reshape(-1, 1)).reshape(-1)
    valid_label = scaler_label.transform(valid_label.to_numpy().reshape(-1, 1)).reshape(-1)
    test_label = scaler_label.transform(test_label.to_numpy().reshape(-1, 1)).reshape(-1)
    
    train_sequence = create_inout_sequences_mul(train_data, train_label, input_window, output_window, number_of_cols)
    train_sequence = train_sequence[:-output_window]

    valid_sequence = create_inout_sequences_mul(valid_data, valid_label, input_window, output_window, number_of_cols)
    valid_sequence = valid_sequence[:-output_window]
    
    test_sequence = create_inout_sequences_mul(test_data, test_label, input_window, output_window, number_of_cols)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), valid_sequence.to(device), test_sequence.to(device)  