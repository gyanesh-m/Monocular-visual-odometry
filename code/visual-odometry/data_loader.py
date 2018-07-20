from glob import glob
from generate_depth_flow import *
from skimage.transform import resize
from skimage.io import imread,imshow,imsave
import os
import math
from subprocess import call
from skimage.restoration import denoise_tv_chambolle
import warnings
from skimage import io
import sys,os
from parser import *
import time
import shutil
#https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# https://stackoverflow.com/questions/24240039/save-numpy-array-as-image-with-high-precision-16-bits-with-scikit-image
# io.use_plugin('freeimage')
def get_ground_6d_poses(p):
	""" For 6dof pose representaion """
	pos = np.array([p[3], p[7], p[11]])
	R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
	angles = rotationMatrixToEulerAngles(R)
	return np.concatenate((pos, angles))

def isRotationMatrix(R):
	""" Checks if a matrix is a valid rotation matrix
		referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
	"""
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

def rotationMatrixToEulerAngles(R):
	""" calculates rotation matrix to euler angles
		referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
	"""
	assert(isRotationMatrix(R))
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6

	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])

def resize_images(path,ht,wd,output_dir=None,ftype='png'):
	"""Resizes the files present in the image directory located by path."""
	files=glob(path+"/*."+ftype)
	if(output_dir is None):
		output_dir=path
	if(not os.path.isdir(output_dir)):
			os.makedirs(output_dir)
	for idx,file_ in enumerate(files):
		im=imread(file_)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			if(im.dtype!='uint8'):
				# Currently the range has not been preserved. Lets' see how it goes.
				# If the model doesn't train well, change this or find a work around 
				# so that the range of input image is preserved while saving it.
				# Currently, input dtype is uint 16 and output d type is uint 8.
				im=resize(im,(ht,wd),anti_aliasing=True)#,preserve_range=True)
			else:
				im=resize(im,(ht,wd),anti_aliasing=True)

			imsave(output_dir+"/"+file_.split("/")[-1],im)
			print("%02.2f"%((idx+1)/len(files)*100)+"%",end="\r")

## Assume optical flow output and depth output is inside the sequence number of the folder
## so  03->Depth,04->Depth,01->Depth
## 01 -> OpticalFlow ,02 -> OpticalFlow ,04 -> OpticalFlow
class Kitty(object):
	def __init__(self,FLAGS,ht=128,wd=384,SEQ=5):
		self.data_dir=FLAGS.data_dir
		self.output_dir=FLAGS.output_dir
		self.pose_dir=FLAGS.pose_dir
		self.trainSq=FLAGS.train_seq
		self.testSq=FLAGS.test_seq
		self.valSq=FLAGS.val_seq
		self.ht=ht
		self.wd=wd
		self.flow_dir=self.output_dir+"/{sq}/"+"OpticalFlow/flo/"
		self.depth_dir=self.output_dir+"/{sq}/"+"Depth/"
		self.timesteps=SEQ
		self.data_paths={"depth_flow":[],"optical_flow":[],"pose":[]}
		self.exist_seq={'train':self.npy_exists(self.trainSq),'test':self.npy_exists(self.testSq),'val':self.npy_exists(self.valSq)}
		self.istrain=FLAGS.train
		self.overlap=FLAGS.overlap
		print(self.exist_seq)
		# assert False
		self.global_id={'train':0,'test':0,'val':0}
	def generate_flow_and_depth(self,path,i_flow,i_depth):
		"""This generates the optical flow and depth images for the resized images. Here path is the 
		complete path including sequence number."""
		base=os.getcwd()
		os.chdir("./../")
		cmd=['python','parser.py','-i',os.path.join(path,"Single"),'-o',os.path.join(path),'-f','2','-m','2']
		print("Generating flow files")
		flo_p=path+"/OpticalFlow/flo/"
		print(os.path.isdir(flo_p))
		print(i_flow)
		if(not i_flow):
			call(cmd)
		else:
			print("Flow files already exists.")
		print(path)
		time.sleep(2)
		print("Generating depth files")
		cmd[-1]='3'
		depth_p=path+"/Depth/"
		print(i_depth)
		if(not i_depth):
			call(cmd)
		else:
			print("Depth files already exists.")
		print(path)
		time.sleep(2)
		os.chdir(base)

	def resize_im(self,typ='train'):
		if(typ=='train'):
			current_sq=self.trainSq
		elif(typ == 'test'):
			current_sq=self.testSq
		elif(typ == 'val'):
			current_sq = self.valSq
		print("Resizing Single images to "+str(self.ht)+"x"+str(self.wd))
		for i,seq_ in enumerate(current_sq):
			seq_=str(seq_)
			if(self.exist_seq[typ][i]['single']):
				print("Single Image already exists for sequence "+seq_+".Skipping its resizing.")
				continue
			output_path=self.output_dir+"/{}/Single/".format(seq_.zfill(2))
			resize_images(self.data_dir+"/sequences/{}/image_2/".format(str(seq_).zfill(2)),self.ht,self.wd,output_path)
		for i,seq_ in enumerate(current_sq):
			seq_=str(seq_)
			i_flow,i_depth=self.exist_seq[typ][i]['flow'],self.exist_seq[typ][i]['depth']
			path=self.output_dir+"/{}/".format(seq_.zfill(2))
			self.generate_flow_and_depth(path,i_flow,i_depth)
		print("Resizing Depth images to "+str(self.ht)+"x"+str(self.wd))
		for i,seq_ in enumerate(current_sq):
			seq_=str(seq_)
			path=self.output_dir+"/{}/Depth/".format(str(seq_).zfill(2))
			try:
				first_image = glob(path+"/*.png")[-1]
				check_shape = imread(first_image).shape
				if(check_shape[0]==self.ht and check_shape[1]==self.wd):
					print("Resized depth sequence already exists for sequence "+seq_+". Skipping its generation.")
					continue
			except Exception as e:
				print(e)
				resize_images(path,self.ht,self.wd)
		
	def get_poses(self, path):
		with open(path) as f:
			poses = np.array([[float(x) for x in line.split()] for line in f])
		return poses
	def generate_data_npy(self,typ='train'):
		if(typ == 'train'):
			Tseq=self.trainSq
		elif(typ == 'test'):
			Tseq=self.testSq
			try:
				shutil.rmtree(self.output_dir+"/test/")
				print("Removed test files.")
			except Exception as e:
				print(e)
				print("Failed to remove test files.")
				# assert False
		elif(typ == 'val'):
			Tseq = self.valSq
			try:
				shutil.rmtree(self.output_dir+"/val/")
				print("Removed validation files.")
			except Exception as e:
				print(e)
				print("Failed to remove validation files.")

			# print("Removing previous generated test files.")
		for i,seq in enumerate(Tseq):
			number_=glob(self.output_dir+"/"+typ+"/*.npz")
			
			if(len(number_)!=0):#self.exist_seq[typ][i]):
				################################
				# Need to improve this consistency check
				print("Data already exists.")
				print("Loading existing dataset.")
				self.global_id[typ]+=len(number_)
				print("Global id number is "+str(self.global_id[typ]))
				time.sleep(2)
				break
			pose_path =  self.pose_dir + 'poses/'+ '%02d.txt' % seq
			flow_path = self.flow_dir.format(sq=str(seq).zfill(2))
			depth_path = self.depth_dir.format(sq=str(seq).zfill(2))
			print(pose_path,flow_path,depth_path)
			pose_seq=self.get_poses(pose_path)
			flow_seq=sorted(glob(flow_path+"*.flo"))
			depth_seq=sorted(glob(depth_path+"*.png"))
			print(len(pose_seq),len(flow_seq),len(depth_seq))
			check=len(pose_seq)!=0 and len(flow_seq)!=0 and len(depth_seq)
			assert check,"Incorrect path, files not found."
			npd=[]
			npf=[]
			for img_ in flow_seq:
				image=readFlow(img_)
				npf.append(image)
			if(not os.path.isdir(self.output_dir+"/"+str(seq).zfill(2)+"/DepthFlow/")):
				os.makedirs(self.output_dir+"/"+str(seq).zfill(2)+"/DepthFlow/")
			for idx in range(len(depth_seq)-1):
				path1=depth_seq[idx]
				path2=depth_seq[idx+1]
				depth_flow=get_depth_flow_and_3dflow((path1,path2),flow_seq[idx],(self.ht,self.wd))
				npd.append(depth_flow.reshape(self.ht,self.wd,1))
				imsave(self.output_dir+str(seq).zfill(2)+"/DepthFlow/"+str(idx).zfill(5)+".png",depth_flow)		
			print(len(npd))
			print(len(npf))
			assert len(npd)==len(npf),"Length of data for depth and optical flow is not same !"
			npd,npf=np.array(npd),np.array(npf)
			if(not os.path.isdir(self.output_dir+"/"+typ+"/")):
				os.makedirs(self.output_dir+"/"+typ+"/")
			print("Saving depth flow and optical flow in numpy.")
			if(self.overlap==1):
				steps=self.timesteps-1
			elif(self.overlap==0):
				steps=1
			
			print('steps',steps)
			time.sleep(2)
			print('npd',len(npd))
			time.sleep(2)
			for idx_d in range(0,len(npd) - self.timesteps+1,steps):
				pose_labels=[]
				for temp_ in range(self.timesteps):
					pose_ = get_ground_6d_poses(pose_seq[idx_d+temp_,:]) - get_ground_6d_poses(pose_seq[idx_d,:])
					pose_labels.append(pose_)
				depth_flo = npd[idx_d:idx_d+self.timesteps]
				opt_flo = npf[idx_d:idx_d+self.timesteps]
				pose_labels=np.array(pose_labels)
				print(depth_flo.shape)
				print(opt_flo.shape)
				print(pose_labels.shape)
				np.savez(self.output_dir+"/"+typ+"/"+str(self.global_id[typ]).zfill(5),depth=depth_flo,optical=opt_flo,pose=pose_labels)
				self.global_id[typ]+=1
				print('global',self.global_id[typ])
			

	def npy_exists(self,sequence):
		if(sequence==None):
			return None
		print(sequence)
		exist_seq=[{'depth':0,'d_flow':0,'flow':0,'single':0} for i in range(len(sequence))]
		for i,seq in enumerate(sequence):
			a=self.output_dir+"/"+str(seq).zfill(2)+"/DepthFlow/"
			b=self.flow_dir.format(sq=str(seq).zfill(2))+"/"
			c=self.output_dir+"/"+str(seq).zfill(2)+"/Single/"
			d=self.depth_dir.format(sq=str(seq).zfill(2))+"/"
			len_a = len(glob(a+"/*.png"))
			len_b = len(glob(b+"/*.flo"))
			len_c = len(glob(c+"/*.png"))
			len_d = len(glob(d+"/*.png"))
			if(len_a != 0):
				exist_seq[i]['d_flow']=1
			if(len_b != 0):
				exist_seq[i]['flow']=1
			if(len_c != 0):
				exist_seq[i]['single']=1
			if(len_d != 0):
				exist_seq[i]['depth']=1
		return exist_seq

	def get_partitions(self):
		partitions={'train':[],'test':[],'val':[]}
		for i in partitions:
			partitions[i]=[str(j).zfill(5) for j in range(self.global_id[i])]
		return partitions