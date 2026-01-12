import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class my_dataset(Dataset):
    def __init__(self,inputs,labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.length = inputs.shape[0]
        return None
    
    def __getitem__(self,index):
        return self.inputs[index,:,:],self.labels[index,:,:]
    
    def __len__(self):
        return self.length

def get_dataloader(batch_size_set):
    with h5py.File("../cooling_data.h5","r") as hdf_file:
        all_inputs = hdf_file["inputs"][:]
        all_labels = hdf_file["labels"][:]
    
    data_length = all_inputs.shape[0]
    data_index = np.arange(data_length)
    np.random.shuffle(data_index)

    train_length = int(0.8*data_length)
    valid_length = int(0.1*data_length)
    
    train_index = data_index[:train_length]
    valid_index = data_index[train_length:train_length+valid_length]
    test_index = data_index[train_length+valid_length:]

    train_inputs = all_inputs[train_index,:,:]
    train_labels = all_labels[train_index,:,:]
    train_dataset = my_dataset(train_inputs,train_labels)

    valid_inputs = all_inputs[valid_index,:,:]
    valid_labels = all_labels[valid_index,:,:]
    valid_dataset = my_dataset(valid_inputs,valid_labels)

    test_inputs = all_inputs[test_index,:,:]
    test_labels = all_labels[test_index,:,:]
    test_dataset = my_dataset(test_inputs,test_labels)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size_set,
                                  shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size_set,
                                  shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size_set,
                                 shuffle=False)
    
    return train_dataloader,valid_dataloader,test_dataloader