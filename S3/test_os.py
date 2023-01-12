import os 
import re 
import torch
import numpy as np 
import torchvision.transforms as transforms

if __name__ == "__main__":


    parent_fodler = "./data/raw/"
    # Create lists to store
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    # Iterate through all the files we want separating them into trainan test
    for root, _, files in os.walk(parent_fodler, topdown=False):
        for name in files:
            if ".npz" in name:
                if bool(re.search(r'_\d', name)):
                    pth_tmp = os.path.join(root, name)
                    x_train.extend(np.load(pth_tmp)["images"])
                    y_train.extend(np.load(pth_tmp)["labels"])

                else:
                    pth_tmp = os.path.join(root, name)
                    x_test.extend(np.load(pth_tmp)["images"])
                    y_test.extend(np.load(pth_tmp)["labels"])

    # Convert them into tensors 
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    # Substract the mean and standard deviation 
    mu_train = x_train.mean()
    std_train = x_train.std()
    mu_test = x_test.mean()
    std_test = x_test.std()

    # Create a Normalize transform
    normalize_transform_train = transforms.Normalize(mu_train, std_train)
    normalize_transform_test = transforms.Normalize(mu_test, std_test)
    
    # Normalize Both tensors 
    x_train = normalize_transform_train(x_train)
    x_test = normalize_transform_test(x_test)

