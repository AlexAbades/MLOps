# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import re

import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

def mnist(input_filepath, output_filepath, batch_size:int=5):
    # specify the folders of the two data folders 
    parent_fodler = input_filepath
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

    # Convert into tensor container for interation simplicity 
    train_set_cont = TensorDataset(x_train, y_train)
    train_set = DataLoader(train_set_cont, shuffle=True, batch_size=batch_size)
    test_set_cont = TensorDataset(x_test, y_test)
    test_set = DataLoader(test_set_cont, shuffle=False, batch_size=batch_size)

    # Save the data 
    torch.save(train_set, f"{output_filepath}/train.pt")
    torch.save(test_set, f"{output_filepath}/test.pt")



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    mnist(input_filepath, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
