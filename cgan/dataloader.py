import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.io import arff
import pandas as pd


def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader

def samplingloader(dataset, format = 'arff'):

    if format == 'csv':
        filename = dataset + '.csv'
        data = pd.read_csv('datasets/' + filename)
    else:
        filename = dataset + '.numeric.' + format
        data= arff.loadarff('datasets/' + filename)
        data = pd.DataFrame(data[0])
        # data = pd.replace()
    
    if dataset == 'vote':
        data = data.replace(b'republican', 0)
        data = data.replace(b'democrat', 1)
        data = data.replace('Class', 'class')

    elif dataset == 'sonar':
        data = data.replace(b'Rock', 0)
        data = data.replace(b'Mine', 1)    

    elif dataset == 'sick':
        data = data.replace(b'negative', 0)
        data = data.replace(b'sick', 1)
    
    elif dataset == 'mushroom':
        data = data.replace(b'e', 0)
        data = data.replace(b'p', 1)
    
    elif dataset == 'labor':
        data = data.replace(b'bad', 0)
        data = data.replace(b'good', 1)
    
    elif dataset == 'kr-vs-kp':
        data = data.replace(b'won', 0)
        data = data.replace(b'nowin', 1)
    
    elif dataset == 'ionosphere':
        data = data.replace(b'b', 0)
        data = data.replace(b'g', 1)
    
    elif dataset == 'hepatitis':
        data = data.replace(b'DIE', 0)
        data = data.replace(b'LIVE', 1)
    
    elif dataset == 'heart-statlog':
        data = data.replace(b'absent', 0)
        data = data.replace(b'present', 1)

    elif dataset == 'diabetes':
        data = data.replace(b'tested_negative', 0)
        data = data.replace(b'tested_positive', 1)

    elif dataset == 'credit-g':
        data = data.replace(b'good', 0)
        data = data.replace(b'bad', 1)

    elif dataset == 'credit-a':
        data = data.replace(b'+', 0)
        data = data.replace(b'-', 1)

    elif dataset == 'colic':
        data = data.replace(b'yes', 0)
        data = data.replace(b'no', 1)

    elif dataset == 'breast-w':
        data = data.replace(b'benign', 0)
        data = data.replace(b'malignant', 1)
    
    elif dataset == 'breast-cancer':
        data = data.replace(b'no-recurrence-events', 0)
        data = data.replace(b'recurrence-events', 1)
    return data

def split_data(data):
    ##
    #   Split test data equally 50% of classes
    #
    ##
    data = data.groupby(['class'])
    a = data.get_group(0)
    b = data.get_group(1)
    a_range = a.shape[0]
    b_range = b.shape[0]
    range_data = min(a_range, b_range) // 2
    train_data_a = a.iloc[range_data:, :]
    train_data_b = b.iloc[range_data:, :]
    test_data_a = a.iloc[:range_data,:]
    test_data_b = b.iloc[:range_data,:]
    train_data = pd.concat([train_data_a, train_data_b])
    test_data = pd.concat([test_data_a, test_data_b])

    all_train_data = train_data.sample(frac=1)
    all_test_data = test_data.sample(frac=1)

    train_data = []
    test_data = []

    for idx in range(all_train_data.shape[0]):
        features = all_train_data.iloc[idx, :-1].values.astype(dtype='float64')
        target = all_train_data.iloc[idx, -1:].values.astype(dtype='float64')
        data = {'features': torch.from_numpy(features),
                 'target': target}
        train_data.append(data)
    
    for idx in range(all_test_data.shape[0]):
        features = all_test_data.iloc[idx, :-1].values.astype(dtype='float64')
        target = all_test_data.iloc[idx, -1:].values.astype(dtype='float64')
        data = {'features': torch.from_numpy(features),
                 'target': target}
        test_data.append(data)

    # train_data = train_data.iloc[:, 0:train_data.shape[1]-1]
    # test_data = test_data.iloc[:, 0:test_data.shape[1]-1]

    train_data = DataLoader(train_data, batch_size=4, shuffle=True)
    test_data = DataLoader(test_data, batch_size=4, shuffle=True)
    features_num = all_train_data.shape[1]-1
    return train_data, test_data, features_num

def load_gan_data(dataset, batch_size):
    data = samplingloader(dataset, 'arff')
    data = data.groupby(['class'])
    a = data.get_group(0)
    b = data.get_group(1)
    a_range = a.shape[0]
    b_range = b.shape[0]
    range_data = min(a_range, b_range)
    data_a = a.iloc[:range_data,:]
    data_b = b.iloc[:range_data,:]
    all_data = pd.concat([data_a, data_b])

    train_data = []

    for idx in range(all_data.shape[0]):
        features = all_data.iloc[idx, :-1].values.astype(dtype='float64')
        target = all_data.iloc[idx, -1:].values.astype(dtype='float64')
        data = {'features': torch.from_numpy(features),
                 'target': target}
        train_data.append(data)
    features_num = all_data.shape[1]-1
    count_data = all_data.shape[0]
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_data, features_num, count_data
