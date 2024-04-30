"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import our own file that has the feature extraction functions
from extract_features_raw import extract_features



#-------------------
# Main script
#-------------------

# INPUT FILEPATH FOR MASK AND IMAGE FOLDER HERE
# Path to the directory containing your images
image_folder_path = r"C:/Users/corny/Documents/Data Science/2. Semester/Projects in DataS/Cute_Cats/FinalProject/images"
mask_folder_path = r"C:/Users/corny/Documents/Data Science/2. Semester/Projects in DataS/Cute_Cats/FinalProject/masks"

#Where is the raw data
file_data = 'metadata.csv'
  
#Where we will store the features
file_features = 'features/features_raw_automatic.csv'

#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data. 
image_id = list(df['img_id'])
label = np.array(df['diagnostic'])

# Here you could decide to filter the data in some way (see task 0)
# For example you can have a file selected_images.csv which stores the IDs of the files you need
is_melanoma = label == 'MEL'
num_images = len(image_id)

#Make array to store features
feature_names = ["image_id", "color_variation_span", "color_variation_amount", "asymmetry",'veil']
num_features = len(feature_names)
features = [] 
 
# Get a list of all files in the directory
file_list = [i for i in os.listdir(image_folder_path) if i.endswith(".png")]
 
# Iterate through each file in the directory
for n,filename in enumerate(file_list):
    # Create the full path to the image file
    image_path = os.path.join(image_folder_path, filename)
    
    # Read the image
    image1 = Image.open(image_path)
    image = np.array(image1.convert("RGB"))
    
 
    #Read in mask as ground truth
    maskname = filename[:-4] + "_mask.png"
    mask_path = os.path.join(mask_folder_path, maskname)
    mask = plt.imread(mask_path)
       
    x = extract_features(image, mask)

    # Store in the variable we created before
    features.append([filename,*x])

        
#Save the image_id used + features to a file   
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  
    
