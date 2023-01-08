import numpy as np
import torch
from torchvision import transforms


def mnist(batch_size:int=10):

    # specify the path where the data is stored 
    files_folder = "./data/"
    # Iterate through all the files to have the path 
    train_files = [f"{files_folder}train_{i}.npz" for i in range(5)]
    # Initialize lists
    x_train = []
    y_train = [] 
    # Load all the data and put it in the same list 
    for files in train_files:
        with np.load(files) as load:
            # Load the images
            x_train.extend(load['images'])
            # Load the labels
            y_train.extend(load['labels'])
    
    # Tramform into tensors 
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    # Specify the test file, in this case just one 
    test_file = "test.npz"
    # Initialize lists 
    with np.load(files_folder + test_file) as load:
        x_test = load["images"]
        y_test = load["labels"]

    # Tranform into tensors 
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    # Defina a Normalization term to deal with the corrupted data 
    # Mean 0.5 std 0.5 in first chanel
    trans = transforms((0.5,), (0.5,))

    # Create a container that stores a collection of tensors in a single object.
    train_set = torch.utils.data.TensorDataset(trans(x_train), y_train)
    # Create a data iterator that provides a way to batch and shuffle the examples in a dataset
    train_container = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create a container and data iterator for test 
    test_set = torch.utils.data.TensorDataset(trans(x_test), y_test)
    test_container = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_container, test_container
