"""
##TODO##
Create separate merged folder for each of the input folders.
"""
from glob import glob
from shutil import copy
import os
import argparse
def run(FLAGS):
	typeof={0:'cpu-multi cut',1:'cpu-moseg-longterm',2:'gpu'}
	with open("./dir-"+typeof[FLAGS.operation]+".txt","r") as f:
		folder_list=[i.split("\n")[0] for i in f.readlines()]
	BASE_DIR=FLAGS.base_dir
	folder_name=FLAGS.folder_name
	try:
		os.makedirs(BASE_DIR+"/merged/"+folder_name+"/SparseSegmentation")
	except Exception as e:
		print(e)
	try:
		os.makedirs(BASE_DIR+"/merged/"+folder_name+"/DenseSegmentation")
	except Exception as e:
		print(e)
	dirs=['SparseSegmentation','DenseSegmentation']
	count=0
	for i,choice in enumerate(dirs):
		if(i==1):
			count=0
		for folder in folder_list:
				current_dir=folder+choice+"/"
				# print(current_dir)
				if(i==0):
					collection_f=sorted(glob(current_dir+"/*.ppm"))
				else:
					collection_f=sorted(glob(current_dir+"/*overlay*.ppm"))
				for file_ in collection_f:
					copy(file_,BASE_DIR+"/merged/"+folder_name+"/"+dirs[i]+"/"+str(count).zfill(5)+".ppm")
					count+=1
					print("Copied "+file_+" to "+BASE_DIR+"/merged/"+folder_name+"/"+dirs[i]+"/"+str(count).zfill(5)+".ppm")