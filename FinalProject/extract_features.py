# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os

# Import packages for image processing
from skimage.transform import rotate
from skimage import morphology
from skimage.io import imread
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import color
from PIL import Image

### ASYMMETRY

# Help function for asymmetry, finds indices of major axis. Given variable should be a mask
def major_axes(mask):
    # Setting counting variable
    column_max = 0
    indice_column_max = None
    degree_max = None

    # Setting up loop for 180 degrees
    for degree in range(0, 180, 10):

        # Rotating the picture
        mask_rotated = rotate(mask, degree)

        # Summing up all columns
        columns_sum = np.sum(mask_rotated, axis = 0)

        # Finding indice of largest
        largest_indice = np.argmax(columns_sum)

        # Checking if larger than current max
        if np.sum(mask_rotated[:, largest_indice]) > column_max:
            column_max = np.sum(mask_rotated[:, largest_indice])
            indice_column_max = largest_indice
            degree_max = degree

    # When done rotating picture we rotate one last time to the rotation of the major axis
    mask_max = rotate(mask, degree_max)

    # We then save the corresponding orthogonal major axis
    indice_row_max = np.argmax(np.sum(mask_max, axis = 1))

    # We return the indices of the max row and max column as well as the max rotated picture
    return indice_column_max, indice_row_max, mask_max

# Calculates the dice_score for a mask by folding it along two major axes. Given variable should be a mask
def asymmetry_score(mask):
    # First we find the major axes and the orientation of the image using our previous function
    major_column, major_row, mask_max = major_axes(mask)

    # We split the columns along the major axis and convert to bool
    column_half1 = mask_max[:, :major_column].astype(bool)
    column_half2 = mask_max[:, major_column:].astype(bool)

    # We make sure they have the same dimensions
    if column_half2.shape[1] > column_half1.shape[1]:
        difference = column_half2.shape[1] - column_half1.shape[1]
        column_add = np.zeros((column_half2.shape[0], difference))
        column_half1 = np.hstack((column_add, column_half1)).astype(bool)

    elif column_half2.shape[1] < column_half1.shape[1]:
        difference = column_half1.shape[1] - column_half2.shape[1]
        column_add = np.zeros((column_half1.shape[0], difference))
        column_half2 = np.hstack((column_half2, column_add)).astype(bool)

    # We "flip" the image around the axis
    column_half2_flipped = column_half2[:, ::-1]

    # We do the same for the rows
    row_half1 = mask_max[:major_row, :].astype(bool)
    row_half2 = mask_max[major_row:, :].astype(bool)

    # Make sure they have the same dimensions
    if row_half2.shape[0] > row_half1.shape[0]:
        difference = row_half2.shape[0] - row_half1.shape[0]
        row_add = np.zeros((difference, row_half2.shape[1]))
        row_half1 = np.vstack((row_add, row_half1)).astype(bool)

    elif row_half1.shape[0] > row_half2.shape[0]:
        difference = row_half1.shape[0] - row_half2.shape[0]
        row_add = np.zeros((difference, row_half1.shape[1]))
        row_half2 = np.vstack((row_half2, row_add)).astype(bool)

    # Flip it
    row_half2_flipped = row_half2[::-1, :]

    # We calculate the overlap
    axes0_overlap = np.sum(column_half1 & column_half2_flipped)
    axes1_overlap = np.sum(row_half1 & row_half2_flipped)
    
    dice = (axes0_overlap*2 + axes1_overlap*2)/(np.sum(mask_max)*2)

    # Return dice
    if dice > 0.88:
        return 0, dice
    elif 0.7 < dice < 0.88:
        return 1, dice
    else:
        return 2, dice

### BLUE-WHITE VEIL

def blue_white_veil(image, mask):

    # Mask with area corresponding to the lesion white and everything else black
    new_mask = mask > 0
    # Apply new mask to image in order to only have the lesion and everything else black
    lesion_masked = image * new_mask[:, :, None]

    # Crop lesion_masked so that the lesion occupies most of the image area
    lesion_coordinates = np.where(mask != 0)
    min_x, max_x = min(lesion_coordinates[0]), max(lesion_coordinates[0])
    min_y, max_y = min(lesion_coordinates[1]), max(lesion_coordinates[1])
    cropped_lesion = lesion_masked[min_x:max_x, min_y:max_y]
   
    # Segment the cropped_lesion given the parameters
    segments = slic(cropped_lesion, n_segments = 10, compactness = 5, sigma = 5, start_label = 1)

    # Count of segments containing blue-white veil
    tot_count = 0
    # Area of segments containing blue-white veil
    tot_area_segment = 0
    # Area of the lesion
    tot_area = np.sum(new_mask)

    # Number of segments genereted by the slic() function on the cropped lesion
    number_segments = np.unique(segments)

    # Iterate through the segments
    for i in number_segments:
        segment = cropped_lesion.copy()
        segment[segments != i] = 0

        # Create an image with the segment and everything else black
        non_black_mask = np.any(segment != [0, 0, 0], axis = 2)

        # Compute the area of the segment
        area = np.sum(non_black_mask)

        # Mean rgb values for the segment
        mean_rgb_values = np.mean(segment[non_black_mask], axis = 0)
        mean_rgb_values_ls = [int(val) for val in np.nan_to_num(mean_rgb_values)]

        # Rnage of acceptable blue RGB values for blue-white veil
        blue = {"min": np.array([50, 50, 106]),
                "max": np.array([128, 200, 200])}

        mean_rgb_values_ls_to_check = np.array([mean_rgb_values_ls])

        # Check if the mean of RGB values is inside the range, if so add 1 to the count of segments containing blue-white veil
        # and add the area of the segment to the area of segments containing blue-white veil
        is_blue = np.all((mean_rgb_values_ls_to_check >= blue["min"]) & (mean_rgb_values_ls_to_check <= blue["max"]))
        if is_blue:
            tot_count += 1
            tot_area_segment += area

    # Compute the proportion of the total area occupied by blue_white veil
    proportion = round((tot_area_segment / tot_area) * 100 if tot_area > 0 else 0, 3)

    # If the proportion is between 20 and 80 return 1 (is blue-white veil)
    if 20 < proportion < 80:
        return 1
    else:
        return 0
    
### COLOR VARIATION

def grayifier(rgb: np.ndarray) -> np.ndarray:
    '''------HELPER FUNCTION------
    Turns a picture into grayscale, returns the picture, reduced to a one dimensional grayscale image, float values'''
    return np.dot(rgb[...,:3],[0.2989, 0.5870, 0.1140])

def crop_n_segment(picture: np.ndarray, mask:np.ndarray) -> np.ndarray:
    '''------HELPER FUNCTION------
    Crops the picture to the mask, gets the superpixels by SLIC and averages the colors of the segments'''
    lesion_coords: tuple = np.where(mask != 0)
    min_x: np.int64 = min(lesion_coords[0])
    max_x: np.int64 = max(lesion_coords[0])
    min_y: np.int64 = min(lesion_coords[1])
    max_y: np.int64 = max(lesion_coords[1])
    cropped_lesion: np.ndarray = picture[min_x:max_x,min_y:max_y]
    segments: np.ndarray = slic(cropped_lesion, n_segments = 20, compactness = 15,sigma = 5, enforce_connectivity = True, convert2lab = True)
    out: np.ndarray = np.empty_like(cropped_lesion)
    out = color.label2rgb(segments,cropped_lesion, kind = "avg")
    return out

def histogrammer(image: np.ndarray) -> np.ndarray:
    '''------HELPER FUNCTION------
    Gets a histogram of the segmented lesion'''
    lesion_feat1:np.ndarray = image[image>0]
    minimum: np.float64 = np.mean(lesion_feat1) - 2*np.std(lesion_feat1)
    maximum: np.float64 = np.mean(lesion_feat1) + 2*np.std(lesion_feat1)
    colr2:np.ndarray = np.histogram(lesion_feat1[(minimum <= lesion_feat1)&(lesion_feat1 <= maximum)], bins = 30, range = (0,255))[0] #a bin is a uniformly increasing value
    return colr2

def get_range(histo:np.ndarray) ->list[int]:
    '''------HELPER FUNCTION------
    Gets the actual interesting facts out of the histogram'''
    for nr,col in enumerate(histo):
        if col !=0:
            numfirst:int = nr
            break
    for revnr,revcol in enumerate(reversed(histo)):
        if revcol !=0:
            numlast: int = revnr
            break
    return [len(histo)-numlast-numfirst, len(histo[histo != 0])]

def color_variation(image, mask):
    '''------Main FUNCTION------
    gets a numerical int value for the color variation of the lesion between 0 and 2'''
    # Workaround: list[np.ndarray] = get_img(os.path.splitext(indv)[0])
    img1: np.ndarray = image
    mask1: np.ndarray = mask

    prepped_img:np.ndarray = crop_n_segment(img1, mask1)

    # Getting the pixel intensity
    plotted_gray:np.ndarray = grayifier(prepped_img.copy())
    historgram:np.ndarray = histogrammer(plotted_gray)
    gray_list:list[int] = get_range(historgram)
    gray_val:int = min([0, 1, 2], key=lambda x:abs(x-((gray_list[0]/gray_list[1]*gray_list[1]/3)-1)))
    return [gray_val, *gray_list]

# Main function to extract features from an image, that calls other functions
def extract_features(image_path, mask_path):

    # Read the image
    image_png = Image.open(image_path)
    image_loaded = np.array(image_png.convert("RGB"))

    # Read in mask as ground truth
    mask_loaded = plt.imread(mask_path)

    # Function for calculating the asymmetry index
    asymmetry_score1, dice_score = asymmetry_score(mask_loaded)

    # Function for calculating the color variation
    color_variation1, gray_span, gray_col = color_variation(image_loaded, mask_loaded)

    # Function for calculating blue_white veil
    blue_white_veil1 = blue_white_veil(image_loaded, mask_loaded)

    return np.array([gray_span, gray_col,dice_score, blue_white_veil1,asymmetry_score1,color_variation1], dtype = np.float16)