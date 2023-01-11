import argparse
import sys
from pathlib import Path

import torch
import click

from model import MyAwesomeModel
from torch import nn 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--ep", default=5, help='epochs to use for training')
def train(lr, ep, optimizer=None, criterion=None):
    print("Training day and night")
    print(f"Learning Rate: {lr}")
    print(f"Number of Epochs: {ep}")

    # TODO: Implement training loop here
    # Create the instance of the model 
    model = MyAwesomeModel()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # if criterion is None:
    criterion = nn.NLLLoss()
    # Load the data
    train_set = torch.load("data/processed/train.pt")
    # set model to train 
    model.train()
    # Create an empty list to save the losses 
    losses = []
    for e in range(ep):
        running_loss = 0
        for images, labels in train_set:
            # We don't have to flatten images as we do it inside the model. 
            images = images.view(images.shape[0], -1)
            # Set gradients to 0
            optimizer.zero_grad()
            # Calculate the output
            logits = model.forward(images)
            # Set the labels to long
            labels = labels.long()
            # Calculate the loss 
            loss = criterion(logits, labels)
            # Perform the backward pass to calculate the gradients 
            loss.backward()
            # Substract the gradient from the weights (perform a step)
            optimizer.step()
            # Store the running_loss 
            running_loss += loss.item()
        loss_tmp = running_loss/len(train_set)
        print(f"Epoch: {e+1}/{ep}\t loss: {loss_tmp:3f}" )
        losses.append(loss_tmp)
    
    # Check if there's a directory for the mnist dataset
    model_path = "models/mnist/"
    model_name = "trained_model.pth"
    model_directory = Path(model_path)
    model_directory.mkdir(parents=True, exist_ok=True)
    save_directory = Path(model_path+model_name)
    # Once the model it's trained, we have to save the model 
    torch.save(model.state_dict(), model_path + model_name)
    if save_directory.exists():
        print(f"Model saved at path: {model_path+model_name}")
    else:
        print("An error occured Model could not be saved")
    
    # Create a figure with matlotlib 
    plt.figure(figsize=(15,10))
    plt.plot(range(1,ep+1), losses)
    plt.xticks(range(1,ep+1))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")

    # Check if there's a directory for the mnist dataset
    model_directory = Path("reports/figures/")
    model_directory.mkdir(parents=True, exist_ok=True)
    # Save model stats 
    plt.savefig("reports/figures/Training_loss.png")
    figure_path = Path("reports/figures/Training_loss.png")
    # Check the files exist
    if figure_path.exists():
        print(f"Figure saved in: {figure_path}")
    else:
        print("An Error occured figure could not be saved")


cli.add_command(train)

if __name__ == "__main__":
    cli()