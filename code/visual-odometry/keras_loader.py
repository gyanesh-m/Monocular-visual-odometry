import keras
import numpy as np
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs,typ='train', output_dir="./data/train/",batch_size=32, time_steps=5 ,dim=(128,384), n_channels=[1,2,6], shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.nd_channels = n_channels[0]
        self.nf_channels = n_channels[1]
        self.np_channels = n_channels[2]
        self.shuffle = shuffle
        self.time_steps=time_steps
        self.output_dir=output_dir
        self.typ=typ
        self.on_epoch_end()

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X_flo = np.empty((self.batch_size, self.time_steps, *self.dim,self.nf_channels))
      X_dflo = np.empty((self.batch_size, self.time_steps, *self.dim,self.nd_channels))
      if(self.typ=='train'):
          Y_po = np.empty((self.batch_size,self.time_steps, self.np_channels))
      else:
          Y_po = None
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          data = np.load(self.output_dir + ID + '.npz')
          X_flo[i,] = data['optical']
          X_dflo[i,] = data['depth']
          if(self.typ=='train'):
              Y_po[i,] = data['pose']
      return {'depth_flow':X_dflo,'optical_flow':X_flo},{'output':Y_po}
    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
      X,Y = self.__data_generation(list_IDs_temp)
      if(self.typ=='train'):
          return X,Y
      else:
          return X