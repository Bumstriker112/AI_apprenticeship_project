# 30 June 2021
# Data cleaner script
# I am uploading data cleaner code as requested.
# Although our model is trained, we might need this for test data 
# separation or for future reference.
# uploaded by : Rishit Malpani (user - bumstriker112)


import os
import shutil

source = r'C:\Desktop\project\data\MYNursingHome' + '\\' #change source and target files as per
target = r'C:\Desktop\project\cleaned_data' + '\\'       # need.

for dirpath, dir, files in os.walk(source):
	if dir:
		main_folders = dir
		for folder in main_folders:
			#create new folders
			path = os.path.join(target, folder)
			os.mkdir(path)

			#copy files
			folder_path = os.path.join(source, folder)
			for subdirpath, subdir, subfiles in os.walk(folder_path):
				for i in range(300): # using 300 for train data, use 100 for test data.
					file_path = os.path.join(folder_path, subfiles[i])
					shutil.copy(file_path, path)

##### end #######
