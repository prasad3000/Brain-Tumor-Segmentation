import sys
import normalizationPipeline
import os

trainf = open('train.txt', 'r')
train_dir = trainf.readlines()

for i in range(int(len(train_dir))):
    print("image" + str(i))
	
    if i % 5 == 4:
        continue
    
    parent_dir = 'G:\\my_study\\Fourth_year_project\\brain_tumor\\BraTS\Auto_focus_me\\data\BRATS2015_Training\\'
    
    direct_mask,_ = train_dir[5 * int(i/5)].split("\n")    
    direct_image,_ = train_dir[i].split("\n")    
    direct_mask = direct_mask + "\mask.nii.gz"
    direct_image_ = os.path.join(parent_dir +direct_image)
    
    pathToMainFolderWithSubjects = ".\\HGG\\"
    subjectsToProcess = os.listdir(pathToMainFolderWithSubjects)
    subjectsToProcess.sort()
 
    saveOutput = True
    prefixToAddToOutp = "_zNorm2StdsMu"
     
    dtypeToSaveOutput = "float32"
    saveNormalizationPlots = True

    lowHighCutoffPercentile = [5., 95.]
    lowHighCutoffTimesTheStd = [3., 3.]
    cutoffAtWholeImgMean = True 
    
    parent_file = 'G:\\my_study\\Fourth_year_project\\brain_tumor\\BraTS\\Auto_focus_me\\data_preprocessing\\data_norm'
    
    normalizationPipeline.do_normalization( parent_file,
				pathToMainFolderWithSubjects,
				subjectsToProcess,
				direct_image_, direct_image, direct_mask,				
				saveOutput,
				prefixToAddToOutp,				
				dtypeToSaveOutput,
				saveNormalizationPlots,								
				lowHighCutoffPercentile, # Can be None
				lowHighCutoffTimesTheStd, # Can be None
				cutoffAtWholeImgMean,
				)


