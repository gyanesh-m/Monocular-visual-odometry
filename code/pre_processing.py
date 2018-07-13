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
	def __init__(self, split_dir, format_name, img_dir=None, model_checkpt=None, model_dir=None,batch=1,output_dir=None,data_dir=None):
		self.data_dir = data_dir
		self.split_dir = split_dir
		self.format = format_name
		self.img_dir=img_dir
		self.model_checkpt=model_checkpt
		self.model_dir=model_dir
		self.batch_size=batch
		self.output_dir=output_dir

	def make_dirs(self,path):
		try:
			os.makedirs(path)
		except Exception as e:
			print(e)

	def extract_images(self):
		"""Returns the sequence of videos in the train/test split file from EGTEA+ dataset
		"""
		#_data contains training and test video file strings from the split file.
		if(self.format==1):
			#EGTEA DATA FORMAT
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
			#For any folder containing many videos.
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
				print(os.path.join(self.data_dir,file_))
				cmd=['ffmpeg','-i',os.path.join(self.data_dir,file_),os.path.join(self.output_dir,"Single",file_.split(".")[0],file_.split(".")[0]+"-%06d.png")]
				call(cmd)
	def call_flow(self,output_dir,current_dir):
		"""Makes call to the flownet2 model to generate flo files. After this, it also generates
		their visualisations. Both of them are saved under separated folders in output dir."""
		self.make_dirs(output_dir)
		self.make_dirs(output_dir+"/flo/")
		self.make_dirs(output_dir+"/png/")
		print(current_dir)
		print(os.path.join(current_dir)+"/")
		cmd=['python', os.path.join(self.model_dir,'main.py'),'--inference', '--model',
			OPTICAL_FLOW_MODEL, '--save_flow', '--inference_dataset', 'ImagesFromFolder', 
			'--inference_dataset_root', os.path.join(current_dir)+"/", '--resume', self.model_checkpt,
			'--batch_size',self.batch_size]
		call(cmd)
		epoch_dir=os.listdir(os.path.join(self.model_dir,'work','inference'))
		print(epoch_dir)
		flow_dir = os.path.join(self.model_dir, 'work', 'inference', epoch_dir[0])
		for flow_files in glob(flow_dir+"/*.flo"):
			call(['./color_flow',flow_files,flow_files[:-3]+'png'])
			shutil.move(flow_files[:-3]+'png',os.path.join(output_dir,"png",flow_files.split("/")[-1][:-3]+"png"))
			shutil.move(flow_files,os.path.join(output_dir,"flo",flow_files.split("/")[-1]))
		shutil.rmtree(flow_dir)
		print("Saved "+str(len(flow_dir))+"Optical flow files at - "+output_dir)

	def generate_optical_flow(self):
		"""Generates the flow files for the images in the dir specified(default Single dir)
		Always specify complete path for the input directory and output directory.
		Relative paths won't work.
		"""
		if(self.format==1):
			#EGTEA format
			for folder in os.listdir(os.path.join(self.img_dir,"Single")):
				print(folder,os.path.isdir(os.path.join(self.img_dir,"Single",folder)))
				if(os.path.isdir(os.path.join(self.img_dir,"Single",folder))):
					output_d=os.path.join(self.img_dir,'OpticalFlow',folder)
					current_dir=os.path.join(self.img_dir,'Single',folder)
					self.call_flow(output_d,current_dir)
		if(self.format==2):
			if(self.img_dir is not None):
			#For any single image folder.
				if(self.output_dir is None):
					self.output_dir=self.img_dir+'/../../OpticalFlow/'+self.img_dir.split("/")[-1]+"/"
				self.output_dir=self.output_dir+"/OpticalFlow/"
				self.call_flow(self.output_dir,self.img_dir+"/")

			elif(self.img_dir is None):
			#For any parent folder containing multiple images folders
				if(self.output_dir is None):
					self.output_dir="./../data/train/"
					self.make_dirs(self.output_dir)
				current=self.output_dir+"/Single/"
				folders=os.listdir(current)
				for folder_ in folders:
					print("Getting optical flow for-"+folder_)
					current_dir=os.path.join(current,folder_)
					output_dir=self.output_dir+"/OpticalFlow/"+folder_
					self.call_flow(output_dir,current_dir)

	def generate_depth(self):
		"""Generates monocular depth image for the images in the data directory."""
		if(self.format==1):
			#For EGTEA data format.
			for folder in os.listdir(os.path.join(self.img_dir,"Single")):
				print("Doing for -",folder)
				if(os.path.isdir(os.path.join(self.img_dir,"Single",folder))):
					out_dir=os.path.join(self.img_dir,"../Depth",folder)
					self.make_dirs(out_dir)
					datasets=os.path.join(self.img_dir,'Single',folder)
					print(datasets,"#"*10)
					cmd_GEO=['python',os.path.join(self.model_dir,'geonet_main.py'),'--mode','test_depth','--dataset_dir',
						os.path.join(self.img_dir,'Single',folder+'/'),'--init_ckpt_file',self.model_checkpt,
						'--batch_size',self.batch_size,"--output_dir",out_dir]
					cmd_sfm=['python',os.path.join(self.model_dir,'test_kitti_depth.py'), '--dataset_dir', os.path.join(self.img_dir,'Single',folder+'/'),'--output_dir',out_dir,'--ckpt_file',self.model_checkpt]
					
					cmd_mega=['python',os.path.join(self.model_dir,'demo.py'),'--img-dir',os.path.join(self.img_dir,'Single',folder+'/'),'--output-dir',out_dir,'--checkpoints_dir',self.model_checkpt]
					call(cmd_mega)
		elif(self.format==2):
			#output dir is parent output directory containg subfolders Single, Depth, Optical Flow, etc.
			if(self.img_dir is not None):
			#For any single image folder
				if(self.output_dir==None):
					self.output_dir=self.img_dir+"./../../Depth/"+self.img_dir.split("/")[-1]+"/"
				self.output_dir+="/Depth/"
				self.make_dirs(self.output_dir)
				cmd_geo=['python',os.path.join(self.model_dir,'geonet_main.py'),'--mode','test_depth','--dataset_dir',
					os.path.join(self.img_dir),'--init_ckpt_file',self.model_checkpt,
					'--batch_size',self.batch_size,"--output_dir",self.output_dir]
				cmd_mega=['python',os.path.join(self.model_dir,'demo.py'),'--img-dir',os.path.join(self.img_dir),"--output-dir",self.output_dir,'--checkpoints_dir',self.model_checkpt]
				call(cmd_mega)
				print("Saved depth files at- " +self.output_dir)
			elif(self.img_dir is None):
			#For any parent folder containing multiple image subfolders
				if(self.output_dir==None):
					self.output_dir="./../data/train/"
				folders=os.listdir(self.output_dir+"/Single/")
				for folder_ in folders:
					print("Getting depth for-"+folder_)
					current_dir=os.path.join(self.output_dir,'Single',folder_)
					output_dir=self.output_dir+"/Depth/"+folder_
					self.make_dirs(output_dir)
					cmd_mega=['python',os.path.join(self.model_dir,'demo.py'),'--img-dir',current_dir,"--output-dir",output_dir,'--checkpoints_dir',self.model_checkpt]
					call(cmd_mega)
					print("Saved depth files at - "+output_dir)