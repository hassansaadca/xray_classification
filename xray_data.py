import copy
import matplotlib.pyplot as plt 
import numpy as np
import os 
import progressbar #pip3 install progressbar2
import random
import sys
import time

from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

test_images_dir = 'data/test'
train_images_dir = 'data/train'
validation_images_dir = 'data/val'


def read_and_rescale_data(data_dir, scale, label_filters, max_count, subset, pn_bacterial):
    """
    Reads the data from the files, performs re-encoding and rescaling
    also filters to only the cases (labels) we are working on.

    Parameters: 
        data_dir (string): The name of a directory in which image data is located
        label_filters (list or array): Data labels to load, should match sub-folder names
        scale (int): The size to scale the loaded image data into a square matrix  
        max_count (int): The maximum data files to load
        subset (str): Either 'EQL' or 'PROP' depending on whether the data that is loaded  
                           per label should be equal or proportional to the total available. 
        pn_bacterial (boolean): Only relevant if PNEUMONIA is inclued in label_filters or no label_filters
                                are included. 
                                    If True, only loads Bacterial Pneumonia data. 
                                    If False, loads all Pneumonia data. 

    Output: 
        Tupel of lists (images, labels) where the former (images) is a list of matrix representations of the 
        scaled image data and the latter (labels) is a list of strings of the corresponding label of each 
        entry in the former (images) based on the sub-folder the image was loaded from. 
    """
    images = []
    labels = []
    
    #Calculate number of images per label to load based on the type of subset specified.
    if subset == 'EQL': 
        cutoffs = subset_equally(data_dir, max_count, label_filters)
    else: 
        cutoffs = subset_proportionately(data_dir, max_count, label_filters)
    
    loaded = {}

    #Crawl the data directory files and sub-directories...
    for dir_name, sub_dir_list, file_list in os.walk(data_dir):

        #Ignore anything in the actual directory itself (all images are in the sub-directories).
        if dir_name == data_dir:
            continue
 
        #Set a progress bar so it's easy to see how much longer files will be loading for.
        pbar = progressbar.ProgressBar()
        for file_name in pbar(file_list):

            #The label will match the sub-folder name (root data foloder ignored above.)
            label = os.path.basename(dir_name)
            
            #Ignore any Mac specific files.
            if '.DS' in file_name: 
                continue
            
            #Give that some number of images from the current label should be loaded... 
            if label in cutoffs: 
                
                #Add label to loaded file dictinoary if it's not there already
                #(To keep track of how may files of this lable have been loaded.)
                if label not in loaded: 
                    loaded[label] = 0
                
                #If enough images with the current lable have been loaded, move on.
                if loaded[label] >= cutoffs[label]:
                    continue 

                #Start actual loading process.
                
                file_path = os.path.join(dir_name, file_name)
                
                #If the bacterial pneumonia only flag is set, skip any non-bacterial pneumonia 
                #image fiels if the current lable is PNEUMONIA.
                if pn_bacterial and label == 'PNEUMONIA' and 'bacteria' not in file_path: 
                    continue

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
                
    #Print a summary of what's been loaded by label.
    for label in loaded: 
        print('%s: %s' % (label, loaded[label]))
    print('Total: %d' % sum(list(loaded.values())))
                    
    return(images, labels)


def subset_equally(data_dir, max_count, label_filters):
    """
    Calculate a the number of images from each label specified that should be 
    loaded in order to load a subset equal to or less than the specified size
    with EQUAL distribution of each specified label. 

    Parameters: 
        data_dir (string): the name of the data directory to load from
        max_count (int): the maximum total number of image files to load
        label_filters (list): a list of labels that should be included in the subset

    Returns: 
        A dictionary where the key is the label and the value is the number of 
        image files to load for that label to meet the subset requirements. 
    """

    #Get count of files by label in the data directory.
    counts = count_files_by_label(data_dir, label_filters)
    
    num_labels = len(counts)
    
    #In case a larger number of images files is necessary than available for 
    #a particular label, set the minimum count to the number of files available 
    #in the data directory for the least common label.
    min_count = min(list(counts.values()))
    
    #Calculate what the total number of files per label should be if the 
    #size of the subset is divided evenly. If no maxinum is specified, default 
    #to the minimum set above, so all labels are loaded to match the least 
    #common label. 
    even = int(max_count / num_labels) if max_count else min_count
    
    #If the least common label does not have enough images fulfill the even 
    #division, default to the minimum set above, so all labels are loaded to 
    #match the least common label. 
    if min_count < even: 
        even = min_count 
    
    #Build the label : number of images dictionary 
    cutoffs = {}
    for label in counts: 
        cutoffs[label] = even
    
    return cutoffs


def subset_proportionately(data_dir, max_count, label_filters):
    """
    Calculate a the number of images from each label specified that should be 
    loaded in order to load a subset equal to or less than the specified size
    with PROPORTIONAL distribution of each specified label comared to the total 
    image files available in the data directory. 

    Parameters: 
        data_dir (string): the name of the data directory to load from
        max_count (int): the maximum total number of image files to load
        label_filters (list): a list of labels that should be included in the subset

    Returns: 
        A dictionary where the key is the label and the value is the number of 
        image files to load for that label to meet the subset requirements. 
    """

    #Get count of files by label in the data directory.
    counts = count_files_by_label(data_dir, label_filters) 
    
    #Find the total number of image files across all labels. 
    total = sum(list(counts.values()))
    
    #Build the label : number of images dictionary
    cutoffs = {}
    for label in counts: 
        #Calculate the proportional of images for the current label 
        proportion = counts[label] / total
        #Calculate the same proportion for the current label given the size of the 
        #subset specified or default to all files for the lable if no max count is specified.
        cutoffs[label] = int(max_count * proportion) if max_count else counts[label]
        
    return cutoffs


def count_files_by_label(data_dir, label_filters): 
    """
    Count the total number of image files by label specified in the data directory. 

    Parameters: 
        data_dir (string): the name of the data directory
        label_filters: a list of the labels to include in the count 

    Output: 
        A dictionary where the key  is the label and the value is the total number 
        of image files for that label in the specified data directory. 
    """
    counts = {}
    for dir_name, sub_dir_list, file_list in os.walk(data_dir):
        if dir_name == data_dir:
            continue

        for file_name in file_list:
            label = os.path.basename(dir_name)    
            
            if label_filters and label not in label_filters: 
                continue
                
            if label not in counts: 
                counts[label] = 0
            
            counts[label] += 1
    
    return counts            
            

def load_data(image_dir, scale, label_filters, max_count, subset, pn_bacterial):
    # read and process images
    images, labels = read_and_rescale_data(image_dir, scale, label_filters, max_count, subset, pn_bacterial)

    X = np.asanyarray(images)
    y = np.asanyarray(labels)
    
    return(X, y.reshape(1, -1)[0])


def load_test(scale=200, label_filters=None, max_count=None, subset='EQL', pn_bacterial=False):
    #load test data
    return(load_data(test_images_dir, scale, label_filters, max_count, subset, pn_bacterial))
  
    
def load_train(scale=200, label_filters=None, max_count=None, subset='EQL', pn_bacterial=False):
    #load train data
    return(load_data(train_images_dir, scale, label_filters, max_count, subset, pn_bacterial))
   
    
def load_val(scale=200, label_filters=None, max_count=None, subset='EQL', pn_bacterial=False):
    #load val data
    return(load_data(validation_images_dir, scale, label_filters, max_count, subset, pn_bacterial))

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
