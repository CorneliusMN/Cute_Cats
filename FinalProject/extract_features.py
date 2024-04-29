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


#Main function to extract features from an image, that calls other functions    
def extract_features(image, mask):
    
    #Function for calculating the asymmetry index
    asymmetry_score = asymmetry_score(mask)

    #Function for calculating the color variation
    color_variation = 

    #Function for calculating blue_white veil
    blue_white_veil = 
    
    return np.array(color_variation, asymmetry_score, blue_white_veil, dtype=np.float16)


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
    
    if dice > 0.88:
        return 0
    elif 0.7 < dice < 0.88:
        return 1
    else:
        return 2
    #return dice
