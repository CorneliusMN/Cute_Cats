"""
FYP project imaging
"""

# Import packages
import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import our own file that has the feature extraction functions
from extract_features import extract_features

#-------------------
# Main script
#-------------------

# INPUT FILEPATH FOR MASK AND IMAGE FOLDER HERE
# Path to the directory containing your images
image_folder_path = r"./images"
mask_folder_path = r"./masks"

# Where is the raw data
file_data = "metadata.csv"

# Where we will store the features
file_features = "features/features_automatic.csv"

# Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data
image_id = list(df["img_id"])
label = np.array(df["diagnostic"])

# Filter for MEL
is_melanoma = label == "MEL"
num_images = len(image_id)

# Make array to store features
feature_names = ["image_id", "color_variation_span", "color_variation_amount", "asymmetry_dicescore", "veil", "color_annotation", "assymetry_annotation"]
num_features = len(feature_names)
features = []

# Get a list of all files in the directory
file_list = [i for i in os.listdir(image_folder_path) if i.endswith(".png")]

# Iterate through each file in the directory
for n,filename in enumerate(file_list):
    print(f"Analysing picture {n}")
    # Create the full path to the image file
    image_path = os.path.join(image_folder_path, filename)

    # Read in mask as ground truth
    maskname = filename[:-4] + "_mask.png"
    mask_path = os.path.join(mask_folder_path, maskname)

    x = extract_features(image_path, mask_path)

    # Store in the variable we created before
    features.append([filename,*x])

# Save the image_id used + features to a file
df_features = pd.DataFrame(features, columns = feature_names)
df_features.to_csv(file_features, index = False)