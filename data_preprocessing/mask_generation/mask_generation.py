import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os

trainf = open('train.txt', 'r')
train_dir = trainf.readlines()

parent_dir = 'G:\\my_study\\Fourth_year_project\\brain_tumor\\BraTS\\Auto_focus_me\\data\\BRATS2015_Training\\'

for i in range(int(len(train_dir)/5)):
    print(i)
    direct,_ = train_dir[5*i].split("\n")    
    # read the images and ground truth
    images0 = nib.load(os.path.join(parent_dir +direct)).get_data()
    images0 = np.expand_dims(images0, axis=0)
        
    direct1,_ = train_dir[5*i + 1].split("\n")    
    # read the images and ground truth
    images1 = nib.load(os.path.join(parent_dir +direct1)).get_data()
    images1 = np.expand_dims(images1, axis=0)
        
    direct2,_ = train_dir[5*i + 2].split("\n")    
    # read the images and ground truth
    images2 = nib.load(os.path.join(parent_dir +direct2)).get_data()
    images2 = np.expand_dims(images2, axis=0)
        
    direct3,_ = train_dir[5*i + 3].split("\n")    
    # read the images and ground truth
    images3 = nib.load(os.path.join(parent_dir +direct3)).get_data()
    images3 = np.expand_dims(images3, axis=0)
        
    images = np.concatenate((images0, images1, images2, images3), axis=0)
    
    image = np.max(images, axis=0)
    image = np.swapaxes(image, 0, 2)
    image[image>0] =1
    image = image.astype(dtype=np.uint8)
    if not os.path.exists(direct):
        os.makedirs(direct)
    image = sitk.GetImageFromArray(image)
    sitk.WriteImage(image, direct + "/mask.nii.gz")