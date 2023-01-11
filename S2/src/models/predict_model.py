import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch import nn 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from pathlib import Path
import os
from torchvision import transforms
import pickle
import numpy as np 
from PIL import Image


@click.group()
def cli():
    pass


@click.command()
@click.argument("--load_model_from", default="", help='Provide a model to load')
@click.argument("--load_data_from", default="", help='Provide a folder to load the data')
def predict(load_model_from, load_data_from):
    print("Evaluating until hitting the ceiling")
    print(f"The model will be load from {load_model_from}")
    print(f"Data will be loaded from {load_data_from}")

    # Initializ the model 
    model = MyAwesomeModel()
    # Load the model parameters
    model.load_state_dict(torch.load(load_model_from))

    # Set model to evaluation 
    model.eval()
    # torch.from_numpy(object)
    transform = transforms.ToTensor()
    # Load the data, there are 3 options, raw data, numpy file pr pickle file. 
    # Check if the directory exists 
    load_data_from = Path(load_data_from)
    assert load_data_from.exists(), "The Selected Directory for loding Data does not exist."

    test_set = []
    for file in load_data_from.iterdir():
        # Pickle Object: should only load one object, 
        if os.path.splitext(file)[1] == ".pkl":
            with open("path/to/file.pkl", "rb") as f:
                loaded_data = np.array(pickle.loads(f.read()))
                test_set = transform(loaded_data)
        # For numpy loadings: Also not prepared for more than one file 
        elif os.path.splitext(file)[1] == ".npy":
            loaded_data = np.load(str(load_data_from))
            test_set = [transform(x.reshape((28, 28, 1))) for x in loaded_data]
        # For raw images 
        else:
            image = Image.open(os.path.join(load_data_from, file))
            # To convert the image to gray scale
            image = image.convert("L")
            test_set.append(transform(image))

    idx = 0
    for im in test_set:
        outputs = model(im)

        out_data = torch.max(outputs, 1)[1].data.numpy()

        for out in out_data:
            print(f"Prediction of input data {idx} : {out}")
            idx += 1




cli.add_command(predict)

if __name__ == "__main__":
    cli()

  