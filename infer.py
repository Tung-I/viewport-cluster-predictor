import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image
from src.model.nets import VPClusterNet 
import sys 
import os
from tqdm import tqdm

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define normalization transform
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(MEAN, STD)

# Load the model
model = VPClusterNet(in_channels=3) 

# # Load pretrained weights only for the ResNet part
# resnet_dict = model.resnet.state_dict()
# resnet_path = '/home/tungi/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
# pretrained_dict = {k: v for k, v in torch.load(resnet_path).items() if k in resnet_dict}
# resnet_dict.update(pretrained_dict)
# model.resnet.load_state_dict(resnet_dict)

model_pth_path = "/home/tungi/viewport-cluster-predictor/models/wmse/train/checkpoints/.pth"
# print the keys of the model
# print(model.state_dict().keys())
# raise Exception("stop")
print(f"Loading model from {model_pth_path}")
checkpoint = torch.load(model_pth_path, map_location=device)
model.load_state_dict(checkpoint['net'])
print("Model loaded")

# model.load_state_dict(torch.load(model_pth_path))  # Load trained model weights
model.to(device)
model.eval()

# Function to process and predict a single image
def process_and_predict(image_path):
    image = Image.open(image_path)
    image = np.array(image).astype(np.float32)
    image = cv2.resize(image, (1024, 512))

    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    image = transforms.functional.to_tensor(image)
    image = normalize(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    return output.cpu().numpy()[0]

# Process frames in a folder and write results to CSV
def process_folder(input_folder, output_csv):
    frame_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('.')[0]))
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        for frame_file in tqdm(frame_files):
            frame_path = os.path.join(input_folder, frame_file)
            output_vector = process_and_predict(frame_path)
            writer.writerow(output_vector)

# Main function to process all video folders
def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for folder in sorted(os.listdir(input_dir)):
        if folder.startswith("Video"):
            input_folder = os.path.join(input_dir, folder)
            output_csv = os.path.join(output_dir, f"{folder}.csv")
            print(f"Processing {input_folder} -> {output_csv}")
            process_folder(input_folder, output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)