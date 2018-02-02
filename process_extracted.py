import os
path = '/Volumes/joe-8tb/wiki-en'

def subdirectory_rename(subdirectory_name, first_extension):
	i = first_extension
	files = os.listdir(subdirectory_name)
	for file in files:
	    os.rename(os.path.join(subdirectory_name, file), os.path.join(subdirectory_name, file[:7]+str(i)[2:5]))    
	return i

def directory_rename(directory_name=path):
	subdirectories = os.listdir(directory_name)
	i = 0.001
	for subdirectory in subdirectories[1:]:
		i = subdirectory_rename(path + '/' + subdirectory, i)
		i += 0.001

## flattened in terminal
# find '/Volumes/joe-8tb/wiki-en' -type f -exec cp {} '/Volumes/joe-8tb/wiki-en-flat' \;


