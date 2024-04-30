#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:53:20 2023

@author: vech
"""
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




#help function for asmmetry, finds indices of major axis. Given variable should be a mask.
def major_axes(mask):
    #setting counting variable
    column_max = 0
    indice_column_max = None
    degree_max = None
    
    #setting up loop for 180 degrees:
    for degree in range(0, 180, 10):
        
        #rotating the picture
        mask_rotated = rotate(mask, degree)
        
        #summing up all columns
        columns_sum = np.sum(mask_rotated, axis = 0)

        #finding indice of largest
        largest_indice = np.argmax(columns_sum)

        #checking if larger than current max:
        if np.sum(mask_rotated[:, largest_indice]) > column_max:
            column_max = np.sum(mask_rotated[:, largest_indice])
            indice_column_max = largest_indice
            degree_max = degree
    
    #when done rotating picture we rotate one last time to the rotation of the major axis
    mask_max = rotate(mask, degree_max)
    
    #we then save the corresponding orthogonal major axis
    indice_row_max = np.argmax(np.sum(mask_max, axis = 1))
    
    #we return the indices of the max row and max column as well as the max rotated picture
    return indice_column_max, indice_row_max, mask_max

#calculates the dice_score for an mask by folding it along two major axes. Given variable should be a mask
def asymmetry_score(mask):
    #first we find the major axes and the orientation of the image using our previous function
    major_column, major_row, mask_max = major_axes(mask)
    
    #we split the columns along the major axis and convert to bool, since float sucks
    column_half1 = mask_max[:, :major_column].astype(bool)
    column_half2 = mask_max[:, major_column:].astype(bool)
    
    #we make sure they have the same dimensions
    if column_half2.shape[1] > column_half1.shape[1]:
        differense = column_half2.shape[1] - column_half1.shape[1]
        column_add = np.zeros((column_half2.shape[0], differense))
        column_half1 = np.hstack((column_add, column_half1)).astype(bool)
    
    elif column_half2.shape[1] < column_half1.shape[1]:
        differense = column_half1.shape[1] - column_half2.shape[1]
        column_add = np.zeros((column_half1.shape[0], differense))
        column_half2 = np.hstack((column_half2, column_add)).astype(bool)
    
    #we "flip" the image around the axis
    column_half2_flipped = column_half2[:, ::-1]
    
    #we do the same for the rows
    row_half1 = mask_max[:major_row, :].astype(bool)
    row_half2 = mask_max[major_row:, :].astype(bool)
    
    #make sure they have the same dimensions
    if row_half2.shape[0] > row_half1.shape[0]:
        differense = row_half2.shape[0] - row_half1.shape[0]
        row_add = np.zeros((differense, row_half2.shape[1]))
        row_half1 = np.vstack((row_add, row_half1)).astype(bool)
        
    elif row_half1.shape[0] > row_half2.shape[0]:
        differense = row_half1.shape[0] - row_half2.shape[0]
        row_add = np.zeros((differense, row_half1.shape[1]))
        row_half2 = np.vstack((row_half2, row_add)).astype(bool)
    
    #and flip it
    row_half2_flipped = row_half2[::-1, :]
    
    #we calculate the overlap
    axes0_overlap = np.sum(column_half1 & column_half2_flipped)
    axes1_overlap = np.sum(row_half1 & row_half2_flipped)
    
    dice = (axes0_overlap*2 + axes1_overlap*2)/(np.sum(mask_max)*2)
    
    return dice




### BLUE-WHITE VEIL

def blue_white_veil(image, mask):

    new_mask = mask > 0
    lesion_masked = image * new_mask[:, :, None]

    lesion_coordinates = np.where(mask != 0)
    min_x, max_x = min(lesion_coordinates[0]), max(lesion_coordinates[0])
    min_y, max_y = min(lesion_coordinates[1]), max(lesion_coordinates[1])
    cropped_lesion = lesion_masked[min_x:max_x, min_y:max_y]
   
    segments = slic(cropped_lesion, n_segments = 10, compactness = 5, sigma = 5, start_label = 1)

    tot_count = 0
    tot_area_segment = 0
    tot_area = np.sum(new_mask)

    number_segments = np.unique(segments)

    for i in number_segments:
        segment = cropped_lesion.copy()
        segment[segments != i] = 0

        non_black_mask = np.any(segment != [0, 0, 0], axis=2)

        area = np.sum(non_black_mask)

        mean_rgb_values = np.mean(segment[non_black_mask], axis = 0)
        mean_rgb_values_ls = [int(val) for val in np.nan_to_num(mean_rgb_values)]

        blue = {"min": np.array([50, 50, 106]),
                "max": np.array([128, 200, 200])}

        mean_rgb_values_ls_to_check = np.array([mean_rgb_values_ls])

        is_blue = np.all((mean_rgb_values_ls_to_check >= blue["min"]) & (mean_rgb_values_ls_to_check <= blue["max"]))
        if is_blue:
            tot_count += 1
            tot_area_segment += area

    proportion = round((tot_area_segment / tot_area) * 100 if tot_area > 0 else 0, 3)

    if 20 < proportion < 80:
        return 1
    else:
        return 0
    
#COLOR VARIATION
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
    segments: np.ndarray = slic(cropped_lesion, n_segments=20, compactness=15,sigma=5, enforce_connectivity=True,convert2lab=True)
    out: np.ndarray = np.empty_like(cropped_lesion)
    out = color.label2rgb(segments,cropped_lesion, kind = "avg")
    return out

def histogrammer(image: np.ndarray) -> np.ndarray:
    '''------HELPER FUNCTION------
    Gets a histogram of the segmented lesion'''
    lesion_feat1:np.ndarray = image[image>0]
    minimum: np.float64 = np.mean(lesion_feat1) - 2*np.std(lesion_feat1)
    maximum: np.float64 = np.mean(lesion_feat1) + 2*np.std(lesion_feat1)
    colr2:np.ndarray = np.histogram(lesion_feat1[(minimum <= lesion_feat1)&(lesion_feat1<=maximum)],bins = 30,range=(0,255))[0] #a bin is a uniformly increasing value
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
    # workaround: list[np.ndarray] = get_img(os.path.splitext(indv)[0])
    img1: np.ndarray = image
    mask1: np.ndarray = mask

    prepped_img:np.ndarray = crop_n_segment(img1, mask1)

    #Getting the pixel intensity
    plotted_gray:np.ndarray = grayifier(prepped_img.copy())
    historgram:np.ndarray = histogrammer(plotted_gray)
    gray_list:list[int] = get_range(historgram)
    #gray_val:int = min([0,1,2], key=lambda x:abs(x-((gray_list[0]/gray_list[1]*gray_list[1]/3)-1)))
    #return gray_val
    return gray_list
#Main function to extract features from an image, that calls other functions    
def extract_features(image, mask):
    
    #Function for calculating the asymmetry index
    asymmetry_score1 = asymmetry_score(mask)

    #Function for calculating the color variation
    color_variation_span, color_variation_amount = color_variation(image, mask)

    #Function for calculating blue_white veil
    blue_white_veil1 = blue_white_veil(image, mask)
    
    return np.array([color_variation_span, color_variation_amount, asymmetry_score1, blue_white_veil1], dtype=np.float16)