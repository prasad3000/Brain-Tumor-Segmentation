import os

def getListOfFiles(dirName, parent_dir):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath, parent_dir)
        else:
            if fullPath[-1] == 'a':
                _, temp = fullPath.split(parent_dir,1)
                allFiles.append(temp)
                
    return allFiles

#write into file
def writeintoFile(file_name, testing_dir, parent_dir):
    listOfFiles = getListOfFiles(testing_dir, parent_dir)
    with open(file_name, 'a') as the_file:
        for file in listOfFiles:
            the_file.write(file + '\n')
    

parent_dir = 'G:\\my_study\\Fourth_year_project\\brain_tumor\\BraTS\\Auto_focus_me\\data\\'
file_name_ =  'BRATS2015_Training';
file_name = file_name_ + '.txt'
testing_dir = parent_dir + file_name_

writeintoFile(file_name, testing_dir, parent_dir)