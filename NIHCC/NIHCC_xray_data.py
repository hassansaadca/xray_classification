import copy
import pandas as pd
import numpy as np
import os 
import progressbar #pip3 install progressbar2
import random
import sys
import time

from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

#Load train and test image list

with  open("data/ChestXRay-NIHCC/train_val_list.txt", "r") as f:
    train_files = list(map(str.strip,f.readlines()))
    print(f'{len(train_files)} images in training set')

with  open("data/ChestXRay-NIHCC/test_list.txt", "r") as f :
    test_files = list(map(str.strip,f.readlines()))
    print(f'{len(test_files)} images in test set')

# load metadata about images
df=  pd.read_csv("data/ChestXRay-NIHCC/Data_Entry_2017_v2020.csv") 

# find unique diagnosis tags 
tagset = df["Finding Labels"].unique()

# remove multiple valued tags
valid_tags = []
train = pd.DataFrame()
test = pd.DataFrame()
for tag in tagset:
    if (tag.find('|')<0 ):
        valid_tags.append(tag)

print("valid tags for dataset: ", valid_tags)

train_rows = df[df["Image Index"].isin(train_files) ]

test_rows = df[df["Image Index"].isin(test_files)]

    
def read_and_rescale_data(image_list, scale, label_filters, max_count, subset):
    """
    Reads the data from the files, performs re-encoding and rescaling
    also filters to only the cases (labels) we are working on.

    Parameters: 
        image_List (list or array): List of image ids to load
        label_filters (list or array): Data labels to load, should match sub-folder names
        scale (int): The size to scale the loaded image data into a square matrix  
        max_count (int): The maximum data files to load
        subset (str): Either 'EQL' or 'PROP' depending on whether the data that is loaded  
                           per label should be equal or proportional to the total available. 
        

    Output: 
        Tupel of lists (images, labels) where the former (images) is a list of matrix representations of the 
        scaled image data and the latter (labels) is a list of strings of the corresponding label of each 
        entry in the former (images) based on the sub-folder the image was loaded from. 
    """
    images = []
    labels = []
    
    
    if subset == 'EQL': 
        cutoffs = subset_equally(image_list, max_count, label_filters)
    else: 
        cutoffs = subset_proportionately(image_list, max_count, label_filters)
    
    loaded = {}
    pbar = progressbar.ProgressBar()
    
    file_list = image_list[image_list["Finding Labels"].isin(label_filters) ]["Image Index"]
    
    for file_name in pbar(file_list):
        
        label = image_list[image_list["Image Index"] == file_name]["Finding Labels"].iloc[0]
        if label in cutoffs: 
                
            if label not in loaded: 
                loaded[label] = 0
            
            if loaded[label] >= cutoffs[label]:
                continue 
                
            file_path = os.path.join("data/ChestXRay-NIHCC/images/", file_name)
                
          # open in with structure to avoid memory leaks
            with Image.open(file_path) as f:
               # copy impage into memory
                image = copy.deepcopy(f)
               # convert image
                image_converted = image.convert("P")
               # rescale
                rescaled = resize(np.array(image_converted), (scale, scale),
                                 anti_aliasing = False).flatten()
                    
               # append to output
                images.append(rescaled)
                    
          # set label to be the index of the label string in the label_filter list
            label_index = label_filters.index(label)
            labels.append(label_index)
                 
            loaded[label] += 1
        else:
            print("dropping ",label)
    
    
    for label in loaded: 
        print('%s: %s' % (label, loaded[label]))
    print('Total: %d' % sum(list(loaded.values())))
                    
    return(images, labels)


def subset_equally(image_list, max_count, label_filters):
    counts = count_files_by_label(image_list, label_filters)
    
    num_labels = len(counts)
    min_count = min(list(counts.values()))
    even = int(max_count / num_labels) if max_count else min_count
    
    if min_count < even: 
        even = min_count 
    
    cutoffs = {}
    for label in counts: 
        cutoffs[label] = even
    
    return cutoffs


def subset_proportionately(image_list, max_count, label_filters):
    counts = count_files_by_label(image_list, label_filters) 
    
    total = sum(list(counts.values()))
    
    cutoffs = {}
    for label in counts: 
        proportion = counts[label] / total
        cutoffs[label] = int(max_count * proportion) if max_count else counts[label]
        
    return cutoffs


def count_files_by_label(image_list, label_filters): 
    counts = {}
    image_labels = image_list[image_list["Finding Labels"].isin(label_filters) ]["Finding Labels"]
    for label in image_labels:
        if label not in counts: 
            counts[label] = 0
        counts[label] += 1
    
    return counts            
            

def load_data(image_list, scale, label_filters, max_count, subset):
    # read and process images
    images, labels = read_and_rescale_data(image_list, scale, label_filters, max_count, subset)

    X = np.asanyarray(images)
    y = np.asanyarray(labels)
    
    return(X, y.reshape(1, -1)[0])


def load_test(scale=200, label_filters=None, max_count=None, subset='EQL'):
    #load test data
    return(load_data(test_rows, scale, label_filters, max_count, subset))
  
    
def load_train(scale=200, label_filters=None, max_count=None, subset='EQL'):
    #load train data
    return(load_data(train_rows, scale, label_filters, max_count, subset))
   
    
def load_val(scale=200, label_filters=None, max_count=None, subset='EQL'):
    #load val data
    print("Validation set not defined for NIHCC ChestXRay dataset")
    return

def find_mean_std(images):
    """
    Generates the mean and standard deviation for a set of images
    
    Parameters:
       images: set of images over which to find the mean
    Output:
        mean: mean of data values over all features
        std: standard deviation of values over all features
    """
    mean = np.mean(images)
    std = np.std(images)
    return (mean,std)

def normalize_images(images,mean,std):
    """
    For a set of images subtracts a mean value, and devides by a standard deviation
    to standardize the range of value for the images
    
    Parameters:
       images: set of images to adjust, note: these do not need to be the images over which the 
       mean and std were calculated
    Output:
       adjusted_images
    """
    adjusted_images = np.empty_like(images)
    for (i,img) in enumerate(images):
        adjusted_image = (img - mean)/std
        #print(adjusted_image.shape)
        adjusted_images[i]= adjusted_image
    return adjusted_images
