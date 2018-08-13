## Multi-modal Egocentric Perception ##

This project aims to tackle the problem of egocentric activity recognition based on the information 
available from video data. It achieves this by the fusion of multi stream convnet
architecture to learn the spatial and temporal features from video data.
The three streams comprise of the following parts:


	*	Single stream
	* 	Optical flow
	*	Depth Flow
	*	Motion segmentation
	*	Visual Odometry
      
### Single frame, Optical Flow, Depth prediction
python parser.py -h
```
usage: parser.py [-h] [-d DIR] [-s SPLIT] [-f {1,2}] [-i IMAGE_DIR]
                 [-md MODEL_DIR] [-mc MODEL_CHECKPT] [--batch_size BATCH_SIZE]
                 [--mode {0,1,2,3}] [--output_dir OUTPUT_DIR]

Generates the training data for specified flows.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Video data directory path.
                        This is the path of the folder which contains subfolders
  -s SPLIT, --split SPLIT
                        Location of Train/Test split file
  -f {1,2}, --format-name {1,2}
                        Specify the format number according to the following mapping-
                        1. EGTEA+ dataset format(Nested folders) 
                        2. Simple Image Folder
  -i IMAGE_DIR, --image-dir IMAGE_DIR
                        Location of image directory
  -md MODEL_DIR, --model-dir MODEL_DIR
                        Location of the optical flow / depth model used.
  -mc MODEL_CHECKPT, --model-checkpt MODEL_CHECKPT
                        Location of optical flow / depth model checkpoint
  --batch_size BATCH_SIZE
                        Mention the batch size for depth and optical flow.
  --mode {0,1,2,3}      Specify the operation to perform.
                        0. Default mode which runs all the operations 
                        1. Simple image
                        2. Optical flow image
                        3. Depth image
  --output_dir OUTPUT_DIR
                        Specified the output location for the code.

```
### Run
```
python parser.py -d <input directory path> --output_dir <output directory path> --mode <any one of 0,1,2,3> --batch_size <number> -f < 2 if it is a directory of videos or images >
```
Incase the mode is specified to be 0, the model directories and checkpoints are not required as they are taken from the default values specified.
In other cases, model directory and checkpoint need to be specified.




### Motion Segmentation

(2 methods )
```
usage: multi-process-work.py [-h] [-o BASE_DIR] [-f FOLDER_NAME] [-m {0,1,2}]
                             [--start START] [--stop STOP] [--splits SPLITS]

Generates the sparse segmentation outputs.

optional arguments:
  -h, --help            show this help message and exit
  -o BASE_DIR, --base_dir BASE_DIR
                        Specify the output directory
  -f FOLDER_NAME, --folder_name FOLDER_NAME
                        Specify the folder name
  -m {0,1,2}, --operation {0,1,2}
                        Specify the mode of operation 
0.cpu-multicut 
1.cpu-moseg-longterm 
2.gpu
  --start START         Specify the first frame number
  --stop STOP           Specify the last frame number
  --splits SPLITS       Specify the times to split the data for parallel
                        Processing.
```

Since this was cpu intensive, so to speedup the process, user can split the data to perform segmentation parallely and then later merge the results.

### Visual Odometry

```
usage: vo-main.py [-h] [-d DATA_DIR] [-p POSE_DIR] [--train {1,2,3}]
                  [--batch BATCH] [--epoch EPOCH]

Train/ Test model based on depth flow and optical flow for visual odometry.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Specify the data directory.
  -p POSE_DIR, --pose_dir POSE_DIR
                        Specify the pose directory.
  --train {1,2,3}       1. for train 
                         2. for test 
                         3. for both train and test. 
  --batch BATCH         Specify the batch size.
  --epoch EPOCH         Specify the epoch size.
```
