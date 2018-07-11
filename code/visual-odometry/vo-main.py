from keras import backend as K
from keras.models import Model
from keras.layers import *
import numpy as np
from keras import regularizers
import argparse
from data_loader import *
BATCH_SIZE = 1
LSTM_HIDDEN_SIZE = 550
LSTM_NUM_LAYERS = 2
# global training time_steps
NUM_TRAIN_STEPS = 2000
TIME_STEPS = 5
##
# 320x640x3 optical flow output
# 384x512 depth image output size 
###

hd,wd=320,640
ho,wo=320,640

class VisualOdo():
	def __init__(self,FLAGS,time_steps=5,checkpoint_dir="./checkpoint/"):
		self.data_dir=FLAGS.data_dir
		self.batch=FLAGS.batch
		self.epoch=FLAGS.epoch
		self.checkpoint_dir=checkpoint_dir
		self.model=None
		self.load_model()

	def load_model(self):
		depth=Input(shape=(TIME_STEPS,hd,wd,1),name='depth_flow')
		opflow=Input(shape=(TIME_STEPS,ho,wo,3),name='optical_flow')
		cnv1=TimeDistributed(Conv2D(64, (7,7),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(depth)
		cnv1_2=TimeDistributed(Conv2D(64, (7,7),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv1)
		cnv2=TimeDistributed(Conv2D(128, (5,5),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(opflow)
		cnv2_2=TimeDistributed(Conv2D(128, (5,5),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv2)
		merged=concatenate([cnv1_2,cnv2_2])
		cnv3=TimeDistributed(Conv2D(256, (5,5),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(merged)
		cnv3_1=TimeDistributed(Conv2D(256, (3,3),strides=1, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv3)
		cnv4=TimeDistributed(Conv2D(512, (3,3),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv3_1)
		cnv4_1=TimeDistributed(Conv2D(512, (3,3),strides=1, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv4)
		cnv5=TimeDistributed(Conv2D(512, (3,3),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv4_1)
		cnv5_1=TimeDistributed(Conv2D(512, (3,3),strides=1, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv5)
		cnv6=TimeDistributed(Conv2D(1024, (3,3),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))(cnv5_1)
		flat=TimeDistributed(Flatten())(cnv6)
		lstm1=LSTM(LSTM_HIDDEN_SIZE,return_sequences=True)(flat)
		lstm2=LSTM(6,return_sequences=True,name='output')(lstm1)
		self.model=Model(inputs=[depth,opflow],outputs=[lstm2])
		self.model.compile(optimizer='adam',
			loss='mean_squared_error',
			metrics=['accuracy'])
		self.model.summary()

	def trial_train(self):
		X={"depth_flow":np.random.rand(50,5,224,224,3),
		"optical_flow":np.random.rand(50,5,224,224,3)}
		Y=np.random.rand(50,5,6)

		self.model.fit(X,Y, batch_size=1, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None)

	def train(self,x,y):
		self.model.fit({'depth_flow':x['depth'],'optical_flow':x['flow']},
						{'output':y},batch_size=self.batch, epochs=self.epoch, verbose=1)

def runner():
	parser = argparse.ArgumentParser(description="Train/ Test model based on depth flow and optical flow for visual odometry." ,formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('-d','--data_dir',type=str,help="Specify the data directory.")
	parser.add_argument('-p','--pose_dir',type=str,help="Specify the pose directory.")
	parser.add_argument('--train',type=int,choices=range(1,4),help="1. for train \n 2. for test \n 3. for both train and test. ")
	parser.add_argument('--batch',type=int,help="Specify the batch size.",default=1)
	parser.add_argument('--epoch',type=int,help="Specify the epoch size.",default=10)
	#ToDo	
	#Add train and test sequence option.
	FLAGS = parser.parse_args()
	vo=VisualOdo(FLAGS)
	kitti_loader=Kitty(FLAGS,trainSq=[1,2,3],testSq=[4])
	X_train,Y_train,X_test,Y_test=kitti_loader.make_train_test()
	if(FLAGS.train==1):
		vo.train(X_train,Y_train)
	elif(FLAGS.train==2):
		vo.test(X_test,Y_test)
	elif(FLAGS.train==3):
		vo.train(X_train,Y_train)
		vo.test(X_test,Y_test)

if __name__=='__main__':
	runner()
