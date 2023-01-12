import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from src.models import MyAwesomeModel

# Initialize Model  
model = MyAwesomeModel()
# Load the model
model.load_state_dict(torch.load("/models/mnist/trained_model.pth"))
# Set model to evaluation 
model.eval()
# Load the training datest. Not in github repo due to the amount of data
train_set = torch.load(" data/processed/train.pt")
# Initialize lists to store features and colors 
features = []
colors = []

# Iterate through the dataset 
for images, labels in train_set:
    # Extract the features at the end of the layer
    out = model.last_layer_features(images)
    # Add features and colors to lists 
    features.extend(out.detach().numpy())
    colors.extend(labels.detach().numpy() * 28 / 255)

# Visualize features in a 2D space using t-SNE to do the dimensionality reduction.
tsne = TSNE(n_components=2).fit_transform(features)

# get x and y components
tx = tsne[:, 0]
ty = tsne[:, 1]

# Normalize the components 
tx = (tx - tx.min()) / (tx.max() - tx.min())
ty = (ty - ty.min()) / (ty.max() - ty.min())

# Create figure 
plt.figure(figsize=(10, 10))
plt.title("Visualizing features at the end of the layer")
plt.scatter(tx, ty, c=colors)
plt.savefig("reports/figures/visual_features.png")
plt.close()