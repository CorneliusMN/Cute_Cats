# Importing necessary packages
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, f1_score
import pickle 


# Read in data with automatically extracted features combined with information from metadata
file_data = 'feautures/features_classifiertraining.csv'
df = pd.read_csv(file_data)
label = np.array(df['diagnostic'])
feature_names = ["color_variation_span", "color_variation_amount", "asymmetry_dicescore", "veil"]

# Make the dataset
x = np.array(df[feature_names])
y =  label == 'MEL'   #now True means melanoma, False means everything else
patient_id = df['patient_id']

# Prepare cross-validation - images from the same patient must always stay together
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)
group_kfold.get_n_splits(x, y, patient_id)


# Different classifiers to test out
classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(11),
    RandomForestClassifier(n_estimators=100, max_depth=None),
    RandomForestClassifier(n_estimators=1000, max_depth=None),
    RandomForestClassifier(n_estimators=1000, max_depth=1),
    RandomForestClassifier(n_estimators=1000, max_depth=3),
    RandomForestClassifier(n_estimators=1000, max_depth=5),
    RandomForestClassifier(n_estimators=10000, max_depth=5),
    RandomForestClassifier(n_estimators=1000, max_depth=10),
    LogisticRegression()
]
num_classifiers = len(classifiers)
classifier_names = ["KN-1", "KN-3", "KN-5", "KN-11", "RF-100-N", "RF-1000-N", "RF-1000-1", "RF-1000-3", "RF-1000-5", "RF-10000-5", "RF-1000-10", "LR"]

# Initialize arrays for evaluators    
acc_val = np.empty([num_folds,num_classifiers])
roc_auc_val = np.empty([num_folds,num_classifiers])
score_val = np.empty([num_folds,num_classifiers])
precision_val = np.empty([num_folds,num_classifiers])
sensitivity_val = np.empty([num_folds,num_classifiers])
specificity_val = np.empty([num_folds,num_classifiers])
f1_val = np.empty([num_folds,num_classifiers])

# Split up data to training and testing groups for each of the folds

for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):
    
    x_train = x[train_index,:]
    y_train = y[train_index]
    x_val = x[val_index,:]
    y_val = y[val_index]
    
    # Run each classifier

    for j, clf in enumerate(classifiers): 
        
        ## Train the classifier
        clf.fit(x_train,y_train)

        ## Evaluate classifier
        
        # Predict probabilities
        y_pred_proba = clf.predict_proba(x_val)[:, 1]
        # Predict binary labels
        y_pred = clf.predict(x_val)
    
        # Calculate accuracy
        acc_val[i,j] = accuracy_score(y_val, y_pred)
        # measures proportion of correctly labelled images from all

        # Calculate area under ROC curve
        roc_auc_val[i,j] = roc_auc_score(y_val, y_pred_proba)
        
        # Calculate precision
        precision_val[i, j] = precision_score(y_val, y_pred, zero_division=0)
        # measures the proportion of true positive predictions out of all positive predictions 
        # made by the classifier. It is calculated as TP / (TP + FP).
        
        # Calculate sensitivity
        sensitivity_val[i, j] = recall_score(y_val, y_pred)
        # measures the proportion of true positive instances that are correctly 
        # identified by the classifier out of all actual positive instances. It is calculated as TP / (TP + FN).
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        specificity_val[i, j] = tn / (tn + fp)
        # it measures the proportion of true negative instances that are correctly 
        # identified by the classifier out of all actual negative instances. It is calculated as TN / (TN + FP).
        
        # Calculate F1 score
        f1_val[i,j] = f1_score(y_val, y_pred)
        # It is calculated as 2 * (precision * sensitivity) / (precision + sensitivity)

   
    
# Average over all folds
average_acc = np.mean(acc_val,axis=0)
average_roc_auc = np.mean(roc_auc_val,axis=0)
average_precision = np.mean(precision_val, axis=0)
average_sensitivity = np.mean(sensitivity_val, axis=0)
average_specificity = np.mean(specificity_val, axis=0)
average_f1 = np.mean(f1_val, axis=0)
   
# Print results from each of the classifiers
for k in range(num_classifiers):
    print(classifier_names[k])
    print('Classifier {} average accuracy- good/all={:.3f} '.format(k+1, average_acc[k]))
    print('Classifier {} average area-under-curve- TP~PF={:.3f} '.format(k+1, average_roc_auc[k]))
    print('Classifier {} average precision- TP / TP+FP={:.3f} '.format(k+1, average_precision[k]))
    print('Classifier {} average sensitivity- TP / TP+FN={:.3f} '.format(k+1, average_sensitivity[k]))
    print('Classifier {} average specificity- TN / TN+FP={:.3f} '.format(k+1, average_specificity[k]))
    print('Classifier {} average F1- 2*precision*sensitivity / precision+sensitivity={:.3f} '.format(k+1, average_f1[k]))
    print('\n')



# Based on the results we have decided to use the Random Forest Classifier with 10,000 estimators and a max depth of 5 
classifier = RandomForestClassifier(n_estimators=10000, max_depth=5)

# It will be tested on external data, so we can try to maximize the use of our available data by training on 
# ALL of x and y
classifier = classifier.fit(x,y)

# Saving the trained classifier
filename = 'CuteCats_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))