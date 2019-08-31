import SimpleITK as sitk
import os
from shutil import copyfile


raw_data = 'BRATS2015_Training.txt'

#copyfile(raw_data, 'mha_data.txt')
copyfile(raw_data, 'nii_train_data.txt')

f=open(raw_data, "r").read().split('\n')[:-1]

with open('mha_train_data.txt','a') as the_file:
    for file in f:
        temp, _ = file.split('.nii')
        temp = temp + '.mha' + '\n'
        the_file.write(temp)

data_mha = open('mha_train_data.txt', 'r')
mha_dir = data_mha.readlines()
data_nii = open('nii_train_data.txt', 'r')
nii_dir = data_nii.readlines()

parent_dir = 'G:\\my_study\\Fourth_year_project\\brain_tumor\\BraTS\\Auto_focus_me\\data\\BRATS2015_Training\\'

for i in range(len(mha_dir)):
    print(i)
    path, _ = os.path.join(parent_dir, mha_dir[i]).split("\n")
    savepath, _ = os.path.join(parent_dir, nii_dir[i]).split("\n")
    img = sitk.ReadImage(path)
    sitk.WriteImage(img, savepath)