import os
import pickle
import faiss
import torch
import torch.nn as nn
import argparse
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Load the pretrained ResNet model
resnet = models.resnet50(pretrained=True)

# Add a linear layer at the end
resnet.fc = nn.Linear(resnet.fc.in_features, 256)

resnet.eval()

# Preprocessing transforms for the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image_embedding(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)

    # Forward pass through the ResNet model
    with torch.no_grad():
        output = resnet(image_tensor)

    # Extract the embedding tensor
    emb_tensor = output.squeeze()
    print(emb_tensor.shape)
    return emb_tensor



def registration_P(directory_path):
    index_file = "img_index.index"
    if os.path.exists(index_file):
        # Load the existing index
        index = faiss.read_index(index_file)
    else:
        # Create a new index
        index = faiss.IndexFlatL2(256)  # Assuming the dimension of your embeddings is 256

    names = []  # List to store the names

    # Get a list of all image files
    file_list = [file_name for file_name in os.listdir(directory_path) if
                 file_name.endswith('.jpg') or file_name.endswith('.png')]

    # Loop over images with a progress bar
    progress_bar = tqdm(file_list, desc="Processing images", unit="image")
    for file_name in progress_bar:
        image_path = os.path.join(directory_path, file_name)

        # Process the image and get the embedding
        emb_tensor = image_embedding(image_path)
        emb_numpy = emb_tensor.detach().numpy().reshape(1, -1)

        # Add the new vector to the index
        index.add(emb_numpy)
        # Add the name to the list
        names.append(file_name)

    # Save the names list to a file
    with open('img_names.pkl', 'wb') as f:
        pickle.dump(names, f)

    # Write the index back to the file
    faiss.write_index(index, index_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script")
    parser.add_argument("--file", type=str, help="Input file path")
    
    args = parser.parse_args()
    
    registration_P(args.file)
    
