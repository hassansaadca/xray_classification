import os 
import progressbar
import pydicom #pip3 install pydicom

test_images_dir = 'data/test'
train_images_dir = 'data/train'
validation_images_dir = 'data/val'

labels = ['NORMAL','PNEUMONIA']

#Make a dicom sub-folder for each label. 
def make_dicom_folders(data_dir): 
    for dir_name, sub_dir_list, file_list in os.walk(data_dir):
        
        if dir_name == data_dir: 
            
            for sub_dir in sub_dir_list: 
                if sub_dir != 'dicom': 
                    dcm_dir = os.path.join(dir_name, sub_dir, 'dicom')
        
                    if not os.path.isdir(dcm_dir): 
                        print('Created %s' % dcm_dir) 
                        os.mkdir(dcm_dir) 
                    else: 
                        print('%s already exists' % dcm_dir) 
            break
            
#Convert image file to dicom.
def convert_dir(data_dir):
    global labels
    
    make_dicom_folders(data_dir)
    
    for dir_name, sub_dir_list, file_list in os.walk(data_dir):
        
        if dir_name == data_dir or 'dicom' in dir_name:
            continue

        dcm_dir = os.path.join(dir_name, 'dicom')
            
        pbar = progressbar.ProgressBar()
        for file_name in pbar(file_list):
            label = os.path.basename(dir_name)   
            
            if label not in labels: 
                continue
                
            dcm_name = file_name.split('.')[0] + '.dcm'
            dcm_path = os.path.join(dcm_dir, dcm_name)
            
            file_path = os.path.join(dir_name, file_name)
                
            convert_command = 'img2dcm %s %s' % (file_path, dcm_path)
            os.system(convert_command)