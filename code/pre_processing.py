#!/usr/bin/env python
import os
from subprocess import call
from glob import glob
import shutil
OPTICAL_FLOW_MODEL='FlowNet2'
DEPTH_MODEL="mega-depth"

class Preprocess(object):
	"""Extracts images from videos for the two formats. Thereafter, it generates images for all the four phases.

	"""
	def __init__(self, data_dir, split_dir, format_name, img_dir=None, model_checkpt=None, model_dir=None,batch=1,output_dir=None):
		self.data_dir = data_dir
		self.split_dir = split_dir
		self.format = format_name
		self.img_dir=img_dir
		self.model_checkpt=model_checkpt
		self.model_dir=model_dir
		self.batch_size=batch
		self.output_dir=output_dir
	
	def extract_images(self):
		"""Returns the sequence of videos in the train/test split file from EGTEA+ dataset
		"""
		#_data contains training and test video file strings from the split file.
		if(self.format==1):
			_data = {"train":[], "test":[]}
			files = [os.path.join(self.split_dir, i) for i in os.listdir(self.split_dir)]
			for file_nameame in files:
				print(file_nameame)
				with open(file_nameame, 'r+') as nfile:
					for line in nfile:
						key_name = file_nameame.split("/")[-1]#extracts the filename
						_data[key_name.split('_')[0]].append(line.strip())
			# Labels contain the train and test data with each video name mapped to its corresponding action label.
			labels = {}
			for key in _data:
				labels[key] = {}
				for val in _data[key]:
					labels[key][val[:-4]] = val.split(" ")[-3]

			for key in _data:
				single_frame_loc = os.path.join(os.getcwd(), "../data/", key, "Single")
				try:
					#Creates folders for the train and test data
					os.makedirs(single_frame_loc)
				except Exception as error:
					print(error)
				for file_line in _data[key]:
					#Extracts the file name
					file_name = '-'.join(file_line.split("-")[:3])
					try:
						#Creates the corresponding label folder inside the train/test folder.
						os.makedirs(os.path.join(single_frame_loc, labels[key][file_line[:-4]]))
					except Exception as e:
						#The folder already exists, so no need to recreate the folder.
						pass
					print("Extracting frames for file "+file_name)
					call(['ffmpeg', '-i', os.path.join(self.data_dir, file_name, ' '.join(file_line.split(" ")[:-3])+'.mp4')
						  , os.path.join(single_frame_loc, labels[key][file_line[:-4]], file_name[:22]+'%06d.png')])
		if(self.format==2):
			files=os.listdir(self.data_dir)
			if(self.output_dir is None):	
				self.output_dir="./../data/train/"
			print(files)
			for file_ in files:
				print(file_.split("."))
				try:
					os.makedirs(self.output_dir+"/Single/"+file_.split(".")[0])
				except Exception as e:
					print(e)
				print("Extracting frames for "+file_)
				cmd=(['ffmpeg','-i',os.path.join(self.data_dir,file_),os.path.join(self.output_dir,file_.split(".")[0],file_.split(".")[0]+"-%06d.png")])
				call(cmd)

	def generate_optical_flow(self):
		"""Generates the flow files for the images in the dir specified(default Single dir)
		"""
		if(self.format==1):
			for folder in os.listdir(os.path.join(self.img_dir,"Single")):
				print(folder,os.path.isdir(os.path.join(self.img_dir,"Single",folder)))
				if(os.path.isdir(os.path.join(self.img_dir,"Single",folder))):
					call(['python', os.path.join(self.model_dir,'main.py'),'--inference', '--model',
					OPTICAL_FLOW_MODEL, '--save_flow', '--inference_dataset', 'ImagesFromFolder', 
					'--inference_dataset_root', os.path.join(self.img_dir,'Single',folder), '--resume', self.model_checkpt,
					'--batch_size',self.batch_size])    
					try:
						os.makedirs(os.path.join(self.img_dir,'OpticalFlow',folder))
					except OSError as e:
						pass
					#Generate the flow images as png and relocate to the train folder
					# Extracts the directories in order of least modified first, i.e., last modified
					# directory is the last element in the list
					epoch_dir=os.listdir(os.path.join(self.model_dir,'work','inference'))
					flow_dir = os.path.join(self.model_dir, 'work', 'inference', epoch_dir[0])
					for flow_files in glob(flow_dir):
						print(flow_dir)
						call(['./color_flow',flow_files,flow_files[:-3]+'.png'])
						os.rename(flow_files[:-3]+'.png',os.path.join(self.img_dir,'OpticalFlow',folder))
					shutil.rmtree(flow_dir)
		if(self.format==2):
			if(self.img_dir is not None):
			#For any general image folder.
				if(self.output_dir is None):
					self.output_dir=self.img_dir+'/../../OpticalFlow/'+self.img_dir.split("/")[-1]+"/"
				call(['python', os.path.join(self.model_dir,'main.py'),'--inference', '--model',
						OPTICAL_FLOW_MODEL, '--save_flow', '--inference_dataset', 'ImagesFromFolder', 
						'--inference_dataset_root', os.path.join(self.img_dir), '--resume', self.model_checkpt,
						'--batch_size',self.batch_size])
				epoch_dir=os.listdir(os.path.join(self.model_dir,'work','inference'))
				print(len(epoch_dir))
				flow_dir = os.path.join(self.model_dir, 'work', 'inference', epoch_dir[0])
				for flow_files in glob(flow_dir):
					print(flow_dir)
					call(['./color_flow',flow_files,flow_files[:-3]+'.png'])
					os.rename(flow_files[:-3]+'.png',os.path.join(self.output_dir))
				shutil.rmtree(flow_dir)

			elif(self.img_dir is None):
			#For any general multiple image folder subfolders
				if(self.output_dir is None):
					self.output_dir="./../data/train/"
				folders=os.listdir(self.output_dir+"/Single/")
				for folder_ in folders:
					print("Getting optical flow for-"+folder_)
					current_dir=os.path.join(self.output_dir,'Single',folder_)
					output_dr=self.output_dir+"/OpticalFlow/"+folder_
					try:
						os.makedirs(output_dr)
					except Exception as e:
						print(e)
					call(['python', os.path.join(self.model_dir,'main.py'),'--inference', '--model',
						OPTICAL_FLOW_MODEL, '--save_flow', '--inference_dataset', 'ImagesFromFolder', 
						'--inference_dataset_root', os.path.join(current_dir), '--resume', self.model_checkpt,
						'--batch_size',self.batch_size])
					epoch_dir=os.listdir(os.path.join(self.model_dir,'work','inference'))
					print(len(epoch_dir))
					flow_dir = os.path.join(self.model_dir, 'work', 'inference', epoch_dir[0])
					for flow_files in glob(flow_dir):
						print(flow_dir)
						call(['./color_flow',flow_files,flow_files[:-3]+'.png'])
						os.rename(flow_files[:-3]+'.png',os.path.join(output_dir))
					shutil.rmtree(flow_dir)
					print("Saved Optical flow files at - "+output_dir)

	def generate_depth(self):
		"""Generates monocular depth image for the images in the data directory."""
		if(self.format==1):
			#For EGTEA data format.
			for folder in os.listdir(os.path.join(self.img_dir,"Single")):
				print("Doing for -",folder)
				if(os.path.isdir(os.path.join(self.img_dir,"Single",folder))):
					try:
						out_dir=os.path.join(self.img_dir,"../Depth",folder)
						os.makedirs(os.path.join(self.img_dir,'Depth',folder))
					except OSError as e:
						print(e)
					datasets=os.path.join(self.img_dir,'Single',folder)
					print(datasets,"#"*10)
					cmd_GEO=['python',os.path.join(self.model_dir,'geonet_main.py'),'--mode','test_depth','--dataset_dir',
						os.path.join(self.img_dir,'Single',folder+'/'),'--init_ckpt_file',self.model_checkpt,
						'--batch_size',self.batch_size,"--output_dir",out_dir]
					cmd_sfm=['python',os.path.join(self.model_dir,'test_kitti_depth.py'), '--dataset_dir', os.path.join(self.img_dir,'Single',folder+'/'),'--output_dir',out_dir,'--ckpt_file',self.model_checkpt]
					
					cmd_mega=['python',os.path.join(self.model_dir,'demo.py'),'--img-dir',os.path.join(self.img_dir,'Single',folder+'/'),'--output-dir',out_dir,'--checkpoints_dir',self.model_checkpt]
					call(cmd_mega)
		elif(self.format==2):
			if(self.img_dir is not None):
			#For any general Image folder format
				if(self.output_dir==None):
					self.output_dir=self.img_dir+"./../../Depth/"+self.img_dir.split("/")[-1]+"/"
				try:
					os.makedirs(self.output_dir)
				except Exception as e:
					print(e)
				cmd_geo=['python',os.path.join(self.model_dir,'geonet_main.py'),'--mode','test_depth','--dataset_dir',
					os.path.join(self.img_dir),'--init_ckpt_file',self.model_checkpt,
					'--batch_size',self.batch_size,"--output_dir",self.output_dir]
				cmd_mega=['python',os.path.join(self.model_dir,'demo.py'),'--img-dir',os.path.join(self.img_dir),"--output-dir",self.output_dir,'--checkpoints_dir',self.model_checkpt]
				call(cmd_mega)
				print("Saved depth files at- " +self.output_dir)
			elif(self.img_dir is None):
			#For any general multiple image folder subfolders
				if(self.output_dir==None):
					self.output_dir="./../data/train/"
				folders=os.listdir(self.output_dir+"/Single/")
				for folder_ in folders:
					print("Getting depth for-"+folder_)
					current_dir=os.path.join(self.output_dir,'Single',folder_)
					output_dr=self.output_dir+"/Depth/"+folder_
					try:
						os.makedirs(output_dr)
					except Exception as e:
						print(e)
					cmd_geo=['python',os.path.join(self.model_dir,'geonet_main.py'),'--mode','test_depth','--dataset_dir',
						os.path.join(self.img_dir),'--init_ckpt_file',self.model_checkpt,
						'--batch_size',self.batch_size,"--output_dir",self.output_dir]
					cmd_mega=['python',os.path.join(self.model_dir,'demo.py'),'--img-dir',current_dir,"--output-dir",output_dr,'--checkpoints_dir',self.model_checkpt]
					call(cmd_mega)
					print("Saved depth files at - "+output_dir)


