import argparse
import sys

import torch
import click

from data import mnist
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
def train(lr, optimizer=None, criterion=None, ep=5):
    print("Training day and night")
    print(lr)
    print(ep)

    # TODO: Implement training loop here
    # Create the instance of the model 
    model = MyAwesomeModel()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # if criterion is None:
    criterion = nn.NLLLoss()
    # Load the data
    train_set, _ = mnist()
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
    
    
    # Once the model it's trained, we have to save the model 
    torch.save(model.state_dict(), "model_checkpoint")

    # Create a figure with matlotlib 
    plt.figure(figsize=(15,10))
    plt.plot(range(1,ep+1), losses)
    plt.xticks(range(1,ep+1))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig("Training_loss.png")
    plt.show()





@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    # Initializ the model 
    model = MyAwesomeModel()
    # Load the model parameters
    model.load_state_dict(torch.load(model_checkpoint))
    # model = torch.load(model_checkpoint)
    # Set model to evaluation 
    model.eval()
    # Set the criterion 
    criterion = nn.NLLLoss()
    # Load the data
    _, test_set = mnist()

    running_accuracy = 0 
    test_lossess = []
    running_acc = 0
    # Set no gradients
    with torch.no_grad():
        testing_loss = 0
        for images, labels in test_set:
            # Calculate the output log_prob
            output = model.forward(images)
            labels = labels.long()
            # Calculate the loss 
            test_loss = criterion(output, labels)
            # Divide the 
            # Add the running loss 
            testing_loss += test_loss.item()
            # Calculate the accuracy with sklearn 
            output_data = torch.max(output, 1)[1].data.numpy()
            running_acc += accuracy_score(labels, output_data)
            # Calculate Accuracy with mean 
            ps = torch.exp(output)
            eq = (labels.data == ps.max(1)[1])
            running_accuracy = eq.type_as(torch.FloatTensor()).mean()
    
    testing_loss /= len(test_set)
    running_accuracy /= len(test_set)
    running_acc /= len(test_set)
    print(f"Test Loss: {testing_loss}")
    print(f"Accuracy Method 1: {running_accuracy*100}%")
    print(f"Accuracy Method 2: {running_acc*100}%")

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  