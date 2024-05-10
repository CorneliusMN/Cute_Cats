# Medical Image Analysis Toolkit

This repository contains a toolkit for medical image analysis. It consists of several Python scripts for processing images, extracting features, training classifiers, and evaluating the performance of the trained classifier.

## Scripts Overview

### extract_features.py:

- The `extract_features` function takes two inputs:
  1. `image`: folder path + file name as a single string
  2. `mask`: folder path + file name as a single string

- It calls separate functions for each of the features.

- Returns a single array of 6 floats representing the features.

### 01_process_images.py:

- Expects folder paths for images and masks to be specified in lines 22 and 23 respectively.

- Loops through all images and masks in the given folders and runs the `extract_features` function on them.

- Returns and saves a CSV file with the image IDs and scores for each of the features.

### 02_train_classifiers.py:

- Uses an edited version of the CSV output from `01_process_images.py` including the diagnoses.

- Splits the data into 5 groups, trains 12 classifiers on each group, and calculates metrics for aggregated results.

- Returns and saves the chosen trained classifier.

### 03_evaluate_classifier.py:

- The `classify` function takes two inputs:
  1. `image`: folder path + file name as a single string
  2. `mask`: folder path + file name as a single string

- Calls the `extract_features` function and uses the trained classifier to predict label based on output.

- Returns and prints the predicted label and probability.

