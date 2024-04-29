"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our own file that has the feature extraction functions
from extract_features import extract_features



#-------------------
# Main script
#-------------------


#Where is the raw data
file_data = 'metadata.csv'
  
#Where we will store the features
file_features = 'features/features_automatic.csv'

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
feature_names = ["color","asymmetry",'veil']
num_features = len(feature_names)
features = np.zeros([num_images,num_features], dtype=np.float16)  


# Path to the directory containing your images
image_folder_path = "features"
mask_folder_path = "masks"
 
# Get a list of all files in the directory
file_list = [i for i in os.listdir(image_folder_path) if i.endswith(".png")]
 
# Iterate through each file in the directory
for n,filename in enumerate(file_list):
    # Create the full path to the image file
    image_path = os.path.join(image_folder_path, filename)
    # Read the image
    image = plt.imread(image_path)[:,:,:3]
 
    #Read in mask as ground truth
    maskname = filename[:-4] + "_mask.png"
    mask_path = os.path.join(mask_folder_path, maskname)
    mask = plt.imread(mask_path)
       
    x = extract_features(image, mask)

    # Store in the variable we created before
    features[n,:] = [filename,*x]

        
#Save the image_id used + features to a file   
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  
    
