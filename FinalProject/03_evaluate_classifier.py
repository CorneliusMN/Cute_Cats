import pickle #for loading trained classifier
import numpy as np
from extract_features import extract_features #our feature extraction



# The function that classifies new images
def classify(img, mask):
    
     #Extract features (only take in first 4 values, as last two are "dumbed down scores")
     x = extract_features(img, mask)[:4]
     x = x.reshape(1, -1)
         
     
     #Load the trained classifier
     classifier = pickle.load(open('CuteCats_classifier.sav', 'rb'))
    
    
     #Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(x)
     pred_prob = classifier.predict_proba(x)
     
     
     print('predicted label is ', pred_label)
     print('predicted probability is ', pred_prob)
     return pred_label, pred_prob
 
    
# The TAs will call the function above in a loop, for external test images/masks

#IMAGE AND MASK SHOULD BE STRINGS FOR FOLDER PATH + FILE NAME, THE EXTRACT_FEATURES WILL READ IT IN FROM THERE
image = 'images/PAT_340_714_68.png'
mask = 'masks/PAT_340_714_68_mask.png'

classify(image, mask)