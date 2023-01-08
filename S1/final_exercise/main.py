import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch import nn 

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--ep", default=5, help='epochs to use for training')
def train(lr, optimaizer=None, criterion=None, ep=5):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    if criterion is None:
        criterion = nn.NLLLoss()
    # Create the instance of the model 
    model = MyAwesomeModel()
    # Load the data
    train_set, _ = mnist()
    # set model to train 
    model.eval()
    for e in ep:
        running_loss = 0
        for images, label in train_set:
            # Flatten images 
            x = images.resize_(images.size()[0], 784)
            # Set gradients to 0
            optimaizer.zero_grad()
            # Calculate the output
            logits = model.forward(x)
            # Calculate the loss 
            loss = criterion(logits, label)
            # Perform the backward pass to calculate the gradients 
            loss.backward()
            # Substract the gradient from the weights (perform a step)
            optimaizer.step()
            # Store the running_loss 
            running_loss += loss.item()
        loss_tmp = loss/len(train_set)
        print(f"Epoch: {e+1/ep}\t loss: {loss_tmp}:3f" )






@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  