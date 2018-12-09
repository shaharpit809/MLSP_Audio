from os import listdir
from os.path import isfile, join
import os
import subprocess
path ='/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Audio_Speech_Actors_01-24/'
path2 = '/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS'

files = []
data = os.listdir(path)
print(data)

for i in data:
	temp_join = join(path,i)
	for j in listdir(temp_join):
		if isfile(join(temp_join,j)):
			files.append(j)
			temp = j.split('.')[0].split('-')
			subprocess.call("mv %s %s" % (join(temp_join,j),join(path2,temp[2])),shell=True)

# # print(files)
# # print(len(files))
# temp2_path = '/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS'
# temp_path = '/home/flash/Documents/IUBBooks/MLSP/Prooject/RAVDESS/Audio_Speech_Actors_01-24/Actor_01' 
# file = [ f for f in listdir(temp_path) if isfile(join(temp_path,f))]

# print(file)

# for  i in file:
# 	temp = i.split('.')[0].split('-')
# 	#temp2 = temp[0].split('-')
# 	print(temp[2])

# 	#os.rename(join(temp_path,i),join(temp_path,temp[2]))
# 	subprocess.call("mv %s %s" % (join(temp_path,i),join(temp2_path,temp[2])),shell=True)
# 	#pathlib.Path(join(temp_path,i)).rename(join(temp_path,temp[2],i))