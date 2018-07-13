from subprocess import call
from glob import glob
import os
import shutil
def move_files(folder_dir,dest,foldername):
	"""Moves files after creation from folder dir to destination inside foldername."""
	multi_path=folder_dir+"/MulticutResults/ldof0.5000008/"
	dest=dest+"/"+foldername+"/Multicut/"
	if(not os.path.isdir(dest)):
		print("created " + dest)
		os.makedirs(dest)
	files=glob(multi_path+"*.ppm")
	print(len(files))
	for file_ in files:
		print(dest+file_.split("/")[-1])
		shutil.move(file_,dest+file_.split("/")[-1])
	print("Deleting "+multi_path)
	shutil.rmtree(multi_path)
def make_bmf(folder_dir):
	"""Creates bmf file for the folder inside data directory."""
	files=sorted(glob(folder_dir+"/*.ppm"))
	print("Generating ppm files for "+str(len(files))+" files from "+folder_dir)
	call(['mogrify', '-format', 'ppm', folder_dir+"/*.png"])
	#extracts file name
	filename=folder_dir.split("/")[-1]
	#takes care of the case when there is a `/` in the end of the path.
	if(len(filename)==0):
		filename=folder_dir.split("/")[-2]
	with open(folder_dir+"/"+filename+".bmf","w+") as bmf:
	    bmf.write(str(len(files))+" 1\n")
	    for i in sorted(files):
	        bmf.write("./"+i.split("/")[-1])
	        bmf.write("\n")

def make_config(iscpu,data_dir):
	"""Modifies the cpu/gpu config for input and output data directories"""
	if(iscpu==1):
		filename="./cpu.cfg"
	else:
		filename="./gpu.cfg"
	with open(filename,"r+") as f:
		lines=f.readlines()
		for i,line in enumerate(lines):
			if('dataDir' in line):
				new_line="s dataDir "+data_dir+"\n"
				lines[i]=new_line
			if('resultDir' in line):
				new_line="s resultDir "+data_dir+"/results\n"
				lines[i]=new_line
			if(iscpu==0):
				if('tracksDir' in line):
					new_line="s tracksDir "+data_dir+"/results\n"
					lines[i]=new_line
		f.seek(0)
		f.write("".join(lines))
		f.truncate()