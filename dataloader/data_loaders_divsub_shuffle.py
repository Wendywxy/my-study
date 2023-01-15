import torch
from torch.utils.data import Dataset
import os
import numpy as np
from glob import *
import math
import random

# split data into training, validation and testing set in ratio 15:4:1
# Change name to load_folds_data for use
def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    #r_p_path = r"util/r_permute_shhs.npy"
    #r_permute = np.load(r_p_path)
    '''
    if "shhs1" in np_data_path:
        r_permute = np.array([23 ,6 ,32 ,9 ,35 ,7 ,33 ,8 ,26 ,13 ,31 ,19 ,20 ,12 ,37 ,4 ,24 ,10 ,27 ,16 ,18 ,22 ,28 ,21 ,36 ,5 ,30 ,29 ,3 ,1 ,34 ,2 ,25 ,39 ,11 ,14 ,41 ,38 ,17 ,40 ,0 ,15 ])
    else:
        r_permute = np.array([2 ,34 ,32 ,48 ,24 ,47,15,44 ,38 ,8 ,30 ,33 ,29,20 ,13 ,1,6,31 ,25 ,16 ,10 ,4 ,45 ,43,28 ,7, 11, 36,46 ,17 ,9 ,49 ,52,26,22, 42 ,3 ,50 ,41 ,0,51,35 ,37 ,23,27,12,19,21,39,14 ,53,5 ,18 ,40 ])
    '''
    '''
    if "shhs1" in np_data_path:
        r_permute = np.array([17 ,29 ,14 ,25 ,39 ,7 ,28 ,8 ,4 ,19 ,31 ,13 ,20 ,12 ,37 ,26 ,32 ,6 ,27 ,16 ,18 ,22 ,10 ,21 ,36 ,5 ,30 ,33 ,3 ,1 ,34 ,2 ,9 ,35 ,11 ,24 ,41 ,38 ,23 ,40 ,0 ,15 ])
    else:
        r_permute = np.array([31, 18, 2, 34, 43, 35, 10, 15, 6, 38, 8, 23, 7, 33, 20, 1, 13, 19, 39, 26, 9, 22, 16, 24, 3, 29, 32, 30, 11, 36, 17, 12, 21, 25, 42, 4, 28, 41, 37, 27, 14, 5, 0, 40])
    '''
    #npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    npzfiles = np.asarray(files, dtype='<U200')
    '''
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    fold_id = 8
    '''
    folds_data = npzfiles
    '''
    subject_files = train_files[fold_id]
    training_files = list(set(npzfiles) - set(subject_files))
    folds_data[fold_id] = [training_files, subject_files]
    '''
    '''
    subject_files = train_files[fold_id:fold_id + 4]
    subject_files = [item for sublist in subject_files for item in sublist]
    #print("subject_files=======:", subject_files)
    #npzfiles = [item for sublist in npzfiles for item in sublist]
    #print("npzfiles=======",npzfiles)
    val_files = train_files[fold_id + 5:fold_id + 9]
    val_files = [item for sublist in val_files for item in sublist]
    #print("val files=======:",val_files)
    training_files = list(set(npzfiles) - set(subject_files) - set(val_files))
    folds_data[fold_id] = [training_files, val_files, subject_files]
    #print("training_files=========",training_files)
    '''
    '''
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    '''
    return folds_data

def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))

    #r_permute = np.array([14, 5, 1, 17, 12, 10, 18, 11, 0, 15, 16, 9, 8, 7, 3, 4, 6, 19, 2, 13])
    #r_permute = np.array([27 ,22 ,16 ,21 ,1 ,23 ,38 ,31 ,24 ,8 ,14, 15 ,3 ,2 ,37 ,28 ,34,9 ,25 ,6 ,19 ,0 ,32 ,36 ,20 ,35 ,29 ,26 ,33 ,30,17,12 ,5 ,4,10 ,18,11 ,7 ,13])

    #folds_data = np.asarray(files, dtype='<U200')[r_permute]
    folds_data = np.asarray(files,dtype='<U200')
    print("folds data = ",folds_data)
    '''
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1]
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs,dtype = object)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    fold_id = 8
    subject_files = train_files[fold_id:fold_id + 4]
    #print("test files",subject_files)
    subject_files = [item for sublist in subject_files for item in sublist]
    files_pairs2 = [item for sublist in files_pairs for item in sublist]
    val_files = train_files[fold_id + 5:fold_id + 9]
    #print("val files",val_files)
    val_files = [item for sublist in val_files for sublist1 in sublist for item in sublist1]
    subject_files = [item for sublist in subject_files for item in sublist]
    training_files = list(set(files_pairs2) - set(subject_files) - set(val_files))
    #print("train files",training_files)
    folds_data[fold_id] = [training_files, val_files, subject_files]
    '''
    '''
    # 15+4+1
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id:fold_id+4]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        if fold_id == n_folds - 1:
            val_files = train_files[0:4]
            val_files = [item for sublist in val_files for sublist1 in sublist for item in sublist1]
            subject_files = [item for sublist in subject_files for item in sublist]
            training_files = list(set(files_pairs2) - set(subject_files) - set(val_files))
        else:
            if fold_id == n_folds - 2:
                val_files1 = train_files[0:3]
                val_files2 = train_files[fold_id + 1]
                val_files1 = [item for sublist in val_files1 for sublist1 in sublist for item in sublist1]
                val_files2 = [item for sublist in val_files2 for sublist1 in sublist for item in sublist1]
                subject_files = [item for sublist in subject_files for item in sublist]
                training_files = list(set(files_pairs2) - set(subject_files) - set(val_files1) - set(val_files2))
                val_files = list(set(files_pairs2) - set(subject_files) - set(training_files))
            else:
                if fold_id == n_folds - 3:
                    val_files1 = train_files[0:2]
                    val_files2 = train_files[fold_id + 1:fold_id + 3]
                    val_files1 = [item for sublist in val_files1 for sublist1 in sublist for item in sublist1]
                    val_files2 = [item for sublist in val_files2 for sublist1 in sublist for item in sublist1]
                    subject_files = [item for sublist in subject_files for item in sublist]
                    training_files = list(set(files_pairs2) - set(subject_files) - set(val_files1) - set(val_files2))
                    val_files = list(set(files_pairs2) - set(subject_files) - set(training_files))
                else:
                    if fold_id == n_folds - 4:
                        val_files1 = train_files[0:1]
                        val_files2 = train_files[fold_id + 1:fold_id + 4]
                        val_files1 = [item for sublist in val_files1 for sublist1 in sublist for item in sublist1]
                        val_files2 = [item for sublist in val_files2 for sublist1 in sublist for item in sublist1]
                        subject_files = [item for sublist in subject_files for item in sublist]
                        training_files = list(
                            set(files_pairs2) - set(subject_files) - set(val_files1) - set(val_files2))
                        val_files = list(set(files_pairs2) - set(subject_files) - set(training_files))
                    else:
                        val_files = train_files[fold_id + 5:fold_id + 9]
                        val_files = [item for sublist in val_files for sublist1 in sublist for item in sublist1]
                        subject_files = [item for sublist in subject_files for item in sublist]
                        training_files = list(set(files_pairs2) - set(subject_files) - set(val_files))
    
        folds_data[fold_id] = [training_files, val_files, subject_files]
    '''
    # print(folds_data[16][0])
    # print(folds_data[16][1])
    # print(folds_data[16][2])
    return folds_data

'''
class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]
        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
'''
class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        #print("len",self.len)
        print("x data",self.x_data.shape)
        print("y data",len(self.y_data))
        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
'''
def data_generator_np(training_files, subject_files, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files)
    print("train data",train_dataset.shape)
    test_dataset = LoadDataset_from_numpy(subject_files)
    print("test data",test_dataset.shape)
    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts
'''


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        print("len of xtrain=%d", len(X_train))
        print("len of ytrain=%d", len(y_train))
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        self.num_channels = min(X_train.shape)
        if X_train.shape.index(self.num_channels) != 1:  # data dim is #samples, seq_len, #channels
            X_train = X_train.permute(0, 2, 1)

        self.len = X_train.shape[0]
        print("x_train shape=%d", X_train.shape[0])
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

'''
#shuffle within subject
def data_generator_np(data_files,data_id, batch_size):
    x = np.load(data_files[0])["x"]
    y = np.load(data_files[0])["y"]
    print("len of y=", y.shape)

    #print("y=", y)
    data_index = np.array([i for i in range(len(y))])
    random.shuffle(data_index)
    x = x[data_index]
    y = y[data_index]
    #print("y shuffled=", y)

    if data_id == 'a':
        len_train = math.ceil(len(y)*0.605) #for edf20
        len_val = int(len(y)*0.1845)#for edf20
    else:
        if data_id == 'b':
            len_train = math.ceil(len(y)*0.5898)#for shhs1
            len_val = int(len(y)*0.192)#for shhs1
        else:
            if data_id == 'c':
                len_train = math.ceil(len(y) * 0.587)  # for shhs2
                len_val = int(len(y) * 0.182)  # for shhs2
    print("len of train=", len_train)
    print("len of val=", len_val)
    X_train = x[0:len_train]
    print("size of x training=", X_train.shape)
    Y_train = y[0:len_train]
    print("size Y_training=", Y_train.shape)
    X_val = x[len_train:len_train + len_val]
    print("size X_val=", X_val.shape)
    Y_val = y[len_train:len_train + len_val]
    print("size Y_val=", Y_val.shape)
    X_test = x[len_train + len_val:]
    print("size X_test=", X_test.shape)
    Y_test = y[len_train + len_val:]
    print("size Y_test=", Y_test.shape)
    for np_file in data_files[1:]:
        x = np.load(np_file)["x"]
        y = np.load(np_file)["y"]

        #print("y=", y)
        data_index = np.array([i for i in range(len(y))])
        random.shuffle(data_index)
        x = x[data_index]
        y = y[data_index]
        #print("y shuffled=", y)

        if data_id == 'a':
            len_train = math.ceil(len(y) * 0.605)  # for edf20
            len_val = int(len(y) * 0.1845)  # for edf20
        else:
            if data_id == 'b':
                len_train = math.ceil(len(y) * 0.5898)  # for shhs1
                len_val = int(len(y) * 0.192)  # for shhs1
            else:
                if data_id == 'c':
                    len_train = math.ceil(len(y) * 0.587)  # for shhs2
                    len_val = int(len(y) * 0.182)  # for shhs2
        X_train = np.vstack((X_train, x[0:len_train]))
        Y_train = np.append(Y_train, y[0:len_train])
        X_val = np.vstack((X_val, x[len_train:len_train + len_val]))
        Y_val = np.append(Y_val, y[len_train:len_train + len_val])
        X_test = np.vstack((X_test, x[len_train + len_val:]))
        Y_test = np.append(Y_test, y[len_train + len_val:])

    print("len of x train=", X_train.shape)
    print("len of y train=", Y_train.shape)
    print("len of x val=", X_val.shape)
    print("len of y val=", Y_val.shape)
    print("len of x test=", X_test.shape)
    print("len of y test=", Y_test.shape)

    train_dataset = dict()
    train_dataset["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    train_dataset["labels"] = torch.from_numpy(Y_train)
    print("train label",train_dataset["labels"])
    val_dataset = dict()
    val_dataset["samples"] = torch.from_numpy(X_val.transpose(0, 2, 1))
    val_dataset["labels"] = torch.from_numpy(Y_val)

    test_dataset = dict()
    test_dataset["samples"] = torch.from_numpy(X_test.transpose(0, 2, 1))
    test_dataset["labels"] = torch.from_numpy(Y_test)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((Y_train, Y_val, Y_test))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader, counts
'''
#shuffle all
def data_generator_np(data_files,data_id, batch_size):
    x = np.load(data_files[0])["x"]
    y = np.load(data_files[0])["y"]
    print("len of y=", y.shape)
    for np_file in data_files[1:]:
        x = np.vstack((x,np.load(np_file)["x"]))
        y = np.append(y,np.load(np_file)["y"])

    print("y=", y)
    data_index = np.array([i for i in range(len(y))])
    random.shuffle(data_index)
    x = x[data_index]
    y = y[data_index]
    print("y shuffled=", y)

    if data_id == 'a':
        len_train = math.ceil(len(y) * 0.60537)  # for edf20
        len_val = int(len(y) * 0.18403)  # for edf20
    else:
        if data_id == 'b':
            len_train = math.ceil(len(y) * 0.590296)  # for shhs1
            len_val = int(len(y) * 0.19138)  # for shhs1
        else:
            if data_id == 'c':
                len_train = math.ceil(len(y) * 0.5873)  # for shhs2
                len_val = int(len(y) * 0.1815)  # for shhs2
    X_train = x[0:len_train]
    Y_train = y[0:len_train]
    X_val = x[len_train:len_train + len_val]
    Y_val = y[len_train:len_train + len_val]
    X_test = x[len_train + len_val:]
    Y_test = y[len_train + len_val:]

    print("len of x train=", X_train.shape)
    print("len of y train=", Y_train.shape)
    print("len of x val=", X_val.shape)
    print("len of y val=", Y_val.shape)
    print("len of x test=", X_test.shape)
    print("len of y test=", Y_test.shape)

    train_dataset = dict()
    train_dataset["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    train_dataset["labels"] = torch.from_numpy(Y_train)
    print("train label",train_dataset["labels"])
    val_dataset = dict()
    val_dataset["samples"] = torch.from_numpy(X_val.transpose(0, 2, 1))
    val_dataset["labels"] = torch.from_numpy(Y_val)

    test_dataset = dict()
    test_dataset["samples"] = torch.from_numpy(X_test.transpose(0, 2, 1))
    test_dataset["labels"] = torch.from_numpy(Y_test)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((Y_train, Y_val, Y_test))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader, counts