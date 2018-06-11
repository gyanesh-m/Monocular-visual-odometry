## Multi-modal Egocentric Perception ##

This project aims to tackle the problem of egocentric activity recognition based on the information 
available from video data. It achieves this by the fusion of multi stream convnet
architecture to learn the spatial and temporal features from video data.
The three streams comprise of the following parts:


	*	Single stream
	* 	Optical flow
    *   Rigid motion flow
      
## Usage


```

usage: pre_processing.py [-h] [-d DIR] [-s SPLIT] [-f FORMAT_NAME]
                         [-i IMAGE_DIR] [-md MODEL_DIR] [-mc MODEL_CHECKPT]
                         [--batch_size BATCH_SIZE] [--mode {0,1,2,3}]

Generates the training data for specified flows.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Video data directory path.
                        This is the path of the folder which contains subfolders
  -s SPLIT, --split SPLIT
                        Location of Train/Test split file
  -f FORMAT_NAME, --format-name FORMAT_NAME
                        Specify the format number according to the following mapping-
                        1. EGTEA+ dataset format 
                        2. Simple Image Folder
  -i IMAGE_DIR, --image-dir IMAGE_DIR
                        Location of image directory
  -md MODEL_DIR, --model-dir MODEL_DIR
                        Location of the optical flow model used.
  -mc MODEL_CHECKPT, --model-checkpt MODEL_CHECKPT
                        Location of optical flow model checkpoint
  --batch_size BATCH_SIZE
                        Mention the batch size for depth and optical flow.
  --mode {0,1,2,3}      Specify the operation to perform.
                        0. Default mode which runs all the operations 
                        1. Simple image
                        2. Optical flow image
                        3. Depth image

```
 