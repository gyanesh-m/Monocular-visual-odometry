from scipy.misc import imread,imresize
from glob import glob
from generate_depth_flow import *
import os
def get_ground_6d_poses(p):
	""" For 6dof pose representaion """
	pos = np.array([p[3], p[7], p[11]])
	R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
	angles = rotationMatrixToEulerAngles(R)
	return np.concatenate((pos, angles))

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
##assume optical flow output and depth output is inside the sequence number of the folder
## so Depth-> 03,04,01
## OpticalFlow -> 01,02,04
class Kitty(object):
	def __init__(self,FLAGS,ht=320,wd=640,trainSq=[1,2,3,4,5,6],testSq=[7,8,9,10]):
		self.data_dir=FLAGS.data_dir
		self.pose_dir=FLAGS.pose_dir
		self.trainSq=trainSq
		self.testSq=testSq
		self.ht=ht
		self.wd=wd
		self.flow_dir=self.data_dir+"/{}/"+"OpticalFlow/"
		self.depth_dir=self.data_dir+"/{}/"+"Depth/"
	def get_poses(self, path):
		with open(path) as f:
			poses = np.array([[float(x) for x in line.split()] for line in f])
		return poses
	def generate_data_npy(self,Tseq,typ='train'):
		for seq in self.Tseq:
			pose_path =  self.pose_dir + 'poses/'+ '%02d.txt' % seq
			flow_path = self.flow_dir + '%02d/' % seq
			depth_path = self.depth_dir + '%02d/' % seq
			pose_seq=self.get_poses(pose_path)
			flow_seq=sorted(glob(flow_path+"*.png"))
			depth_seq=sorted(glob(depth_path+"*.png"))
			npd=[]
			npf=[]
			for img_ in flow_seq:
				image=imread(img_)
				image_=imread.resize(image,(self.ht,self.wd,-1))
				npf.append(image_)
			for idx in range(len(depth_seq)-1):
				path1=depth_seq[idx]
				path2=depth_seq[idx+1]
				depth_flow=get_depth_flow_and_3dflow((path1,path2),flow_seq[idx],(self.ht,self.wd))
				npd.append(depth_flow)
			final_npd=[]
			final_npf=[]
			assert len(npd)==len(npf),"Length of data for depth and optical flow is not same !"
			npd,npf=np.array(npd),np.array(npf)
			npp=[]
			for idx_d in range(0,len(npd)-self.timesteps+1,self.timesteps):
				final_npd.append(npd[idx_d:idx_d+self.timesteps])
				final_npf.append(npf[idx_d:idx_d+self.timesteps])
				pose_labels=[]
				for temp_ in range(self.timesteps):
					pose_ = get_ground_6d_poses(pose_seq[idx_d+temp_,:]) - get_ground_6d_poses(pose_seq[idx_d,:])
					pose_labels.append(pose_)
				npp.append(pose_labels)
			print("Saving depth flow and optical flow in numpy.")
			try:
				os.makedirs(self.depth_dir.format(typ)+"npy/")
			except Exception as e:
				print(e)
			try:
				os.makedirs(self.flow_dir.format(typ)+"npy/")
			except Exception as e:
				print(e)
			try:
				os.makedirs(self.data_dir+typ+"/pose/"+"npy/")
			except Exception as e:
				print(e)

			np.save(self.depth_dir.format(typ)+"/npy/DepthSequence-"+str(seq),np.array(final_npd))
			np.save(self.flow_dir.format(typ)+"/npy/FlowSequence-"+str(seq),np.array(final_npf))
			np.save(self.data_dir+typ+"/pose/npy/PoseSequence-"+str(seq),np.array(npp))
			return {'X':{'depth':np.array(final_npd),
					'flow':np.array(final_npf)},'Y':np.array(npp)}
	def make_train_test(self):
		XY_tr=generate_data_npy(self.trainSq,'train')
		XY_tst=generate_data_npy(self.testSq,'test')
		return(XY_tr['X'],XY_tr['Y'],XY_tst['X'],XY_tst['Y'])
