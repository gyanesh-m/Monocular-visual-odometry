from keras import backend as K
from keras.models import Model
from keras import models
from keras.layers import *
import numpy as np
from keras import regularizers
import argparse
import keras
from keras.callbacks import *
from data_loader import *
from keras_loader import DataGenerator
LSTM_HIDDEN_SIZE = 100
LSTM_NUM_LAYERS = 2
TIME_STEPS = 5
##
# 128x384x3 optical flow output
# 384x512 depth image output size 
###

# hd,wd=128,384
# ho,wo=128,384
hd,wd=64,192
ho,wo=64,192
class VisualOdo():
    def __init__(self,FLAGS,checkpoint_dir,time_steps=5):
        self.data_dir=FLAGS.data_dir
        self.batch=FLAGS.batch
        self.epoch=FLAGS.epoch
        self.checkpoint_dir=checkpoint_dir
        self.output_dir=FLAGS.output_dir
        self.model=None
        self.epoch=FLAGS.epoch
        self.checkpoint_dir=checkpoint_dir
        self.mc=ModelCheckpoint(checkpoint_dir+"/weights.hdf5 {epoch:02d}-loss{loss:.2f}", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.es=EarlyStopping(monitor='loss', min_delta=1, patience=15, verbose=1, mode='auto')
        self.tb=TensorBoard(log_dir=self.output_dir+'/logs',  batch_size=self.batch, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        self.lr=ReduceLROnPlateau(monitor='loss', factor=0.95, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        if(FLAGS.train==1):
            if(FLAGS.load_checkpoint==1):
                self.load_saved_model()
            else:
                self.load_model()
        elif(FLAGS.train==0):
            self.load_saved_model()
            self.fname='_'.join(map(str,FLAGS.test_seq))+'output'
        #Contains the id numbers
        self.list_ids=[]


    def load_model(self):
        depth=Input(shape=(TIME_STEPS,hd,wd,1),name='depth_flow')
        opflow=Input(shape=(TIME_STEPS,ho,wo,2),name='optical_flow')
        cnv1=TimeDistributed(Conv2D(64, (7,7),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(depth)
        cnv1_2=TimeDistributed(Conv2D(64, (7,7),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv1)
        cnv2=TimeDistributed(Conv2D(128, (5,5),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(opflow)
        cnv2_2=TimeDistributed(Conv2D(128, (5,5),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv2)
        merged=concatenate([cnv1_2,cnv2_2])
        cnv3=TimeDistributed(Conv2D(256, (5,5),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(merged)
        cnv3_1=TimeDistributed(Conv2D(256, (3,3),strides=1, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv3)
        cnv4=TimeDistributed(Conv2D(512, (3,3),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv3_1)
        cnv4_1=TimeDistributed(Conv2D(512, (3,3),strides=1, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv4)
        cnv5=TimeDistributed(Conv2D(512, (3,3),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv4_1)
        cnv5_1=TimeDistributed(Conv2D(512, (3,3),strides=1, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv5)
        cnv6=TimeDistributed(Conv2D(1024, (3,3),strides=2, padding='same',dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))(cnv5_1)
        drp_1=TimeDistributed(Dropout(0.5))(cnv6)
        flat=TimeDistributed(Flatten())(drp_1)
        lstm1=LSTM(LSTM_HIDDEN_SIZE,return_sequences=True)(flat)
        lstm2=LSTM(6,return_sequences=True,name='output')(lstm1)
        self.model=Model(inputs=[depth,opflow],outputs=[lstm2])
        adm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(optimizer=adm,
            loss=self.loss_modified)
        self.model.summary()
        if(not os.path.isdir(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
    
    def load_saved_model(self,choice=0):
        files=os.listdir(os.path.join(self.checkpoint_dir))
        print(files[1].split("loss"))
        time.sleep(2)
        file_name={float(i.split('loss')[1]):i for i  in files if '.hdf5' in i}
        file_name_ = sorted([float(i.split('loss')[1]) for i in files])
        print("Select the saved model to use -")
        for i,j in enumerate(file_name_[:25]):
            print(str(i+1)+"-"+str(file_name[j]))
        choice=input()
        print("#"*5+"  Using saved model-"+file_name[file_name_[int(choice)-1]]+"  "+"#"*5)
        model=models.load_model(os.path.join(self.checkpoint_dir,file_name[file_name_[int(choice)-1]]))
        print("#"*5+"  Model Loaded  "+"#"*5)
        self.model=model

    def train(self,x,y):
        self.model.fit({'depth_flow':x['depth'].reshape(-1,TIME_STEPS,hd,wd,1),'optical_flow':x['flow']},
                        {'output':y},batch_size=self.batch, epochs=self.epoch, verbose=1,callbacks=[self.mc,self.tb,self.es,self.lr])
    def train_generator(self,gen,vgen):
        self.model.fit_generator(generator=gen,use_multiprocessing=True,workers=6,callbacks=[self.mc,self.tb,self.es,self.lr],epochs=self.epoch,validation_data=vgen)
        return None
    def test_generator(self,gen):
        self.outputs=[]
        self.outputs.append(self.model.predict_generator(generator=gen,use_multiprocessing=True,workers=6))
        print(len(self.outputs))
        self.outputs=np.array(self.outputs)
        print(self.outputs.shape)
        print("Saving outputs.")
        if(not os.path.isdir(self.output_dir+'/results/')):
            os.makedirs(self.output_dir+"/results/")
        np.save(self.output_dir+"/results/"+self.fname,self.outputs)

    def loss_modified(self,y_true,y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        time.sleep(5)
        return K.mean(K.square(y_true[:,:,:3]-y_pred[:,:,:3])+100*K.square(y_true[:,:,3:]-y_pred[:,:,3:]),axis=-1)
def runner():
    parser = argparse.ArgumentParser(description="Train/ Test model based on depth flow and optical flow for visual odometry." ,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d','--data_dir',type=str,help="Specify the data directory for kitti folder containing folder sequences.")
    parser.add_argument('-p','--pose_dir',type=str,help="Specify the pose directory.")
    parser.add_argument('-o','--output_dir',type=str,help="Specific the output directory.")
    parser.add_argument('--train',type=int,choices=range(2),help="0. for test \n 1. for train .")
    parser.add_argument('--batch',type=int,help="Specify the batch size.",default=1)
    parser.add_argument('--epoch',type=int,help="Specify the epoch size.",default=10)
    parser.add_argument("--train_seq",type=int,nargs='+',help="Specify the train sequence(s) separated by spaces.")
    parser.add_argument("--test_seq",type=int,nargs="+",help="Specify the test sequence(s) separated by spaces.")
    parser.add_argument("--val_seq",type=int,nargs='+',help="Specify the validation sequence(s) separated by spaces.")
    parser.add_argument("--checkpoint_dir",type=str,help="Specify the location to save checkpoints. Default is inside checkpoints folder in output dir.")
    parser.add_argument("-op","--overlap",type=int , choices=range(2),default=1,help="Decides whether to test the output on overlapping sequences(1) or to test on non-overlapping sequences(0).")
    parser.add_argument("--load_checkpoint",type=int, choices=range(2),default=0, help="Loads last saved checkpoint and continues from there.")

    FLAGS = parser.parse_args()
    if(FLAGS.checkpoint_dir is None):
        checkpoint_d = FLAGS.output_dir+"/checkpoint/"
    else:
        checkpoint_d = FLAGS.checkpoint_dir+"/checkpoint/"
    if(FLAGS.train==1):
        current_op = 'train'
        shuffle=1
    else:
        current_op = 'test'
        shuffle=0

    vo=VisualOdo(FLAGS,checkpoint_d)
    kitti_loader=Kitty(FLAGS,SEQ=TIME_STEPS,ht=hd,wd=wd)
    kitti_loader.resize_im(current_op)
    if(current_op=='train'):
        kitti_loader.resize_im('val')
    
    print("Generating "+current_op+" data.")
    kitti_loader.generate_data_npy(typ=current_op)
    if(current_op=='train'):
        kitti_loader.generate_data_npy(typ='val')
    params={'output_dir':FLAGS.output_dir+'/'+current_op+'/',
            'batch_size':FLAGS.batch,
            'time_steps':TIME_STEPS,
            'dim':(hd,wd),
            'n_channels':[1,2,6],
            'shuffle':shuffle,
            'typ':current_op}

    #train and test partition dictionary.
    partitions=kitti_loader.get_partitions()
    print(partitions['train'])
    t_generator=DataGenerator(partitions[current_op],**params)
    if(FLAGS.train==1):
        print("Starting training session.")
        params_1=params
        params_1['output_dir']=FLAGS.output_dir+"/val/"
        v_generator=DataGenerator(partitions['val'],**params)
        vo.train_generator(t_generator,v_generator) 

    elif(FLAGS.train==0):
        vo.test_generator(t_generator)

if __name__=='__main__':
    runner()