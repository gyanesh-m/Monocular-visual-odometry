## Multi-modal Egocentric Perception ##

This project aims to tackle the problem of egocentric activity recognition based on the information 
available from video data. It achieves this by the fusion of multi stream convnet
architecture to learn the spatial and temporal features from video data.
The three streams comprise of the following parts:


  * Single stream
  * Optical flow
  * Depth
  * Visual Odometry

### Dependencies

To install the dependencies, run 

``` 
pip install -r requirements.txt
```

Some other repo which needs to be downloaded locally are:
1. **FlowNet2** [link](https://github.com/NVIDIA/flownet2-pytorch)
2. **MegaDepth** [link](https://github.com/lixx2938/MegaDepth)

I have made few changes to them so I have included them in the utils folder.

### Pretrained models

Pretrained model for mega depth can be found [here](http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth) and for FlowNet2.0, it can be found [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing).


### Code Structure

```
.
├── code
│   ├── color_flow
│   ├── parser.py
│   ├── Pipeline.sh
│   ├── pre_processing.py
│   ├── requirements.txt
│   └── visual-odometry
│       └── odometry2d.py
├── data
│   └── split
│       └── egtea
├── googlecl-pylint.rc
├── README.md
├── requirements.txt
└── utils
    ├── flownet2-pytorch
    └── MegaDepth
```
      
### Single frame, Optical Flow, Depth prediction


python parser.py -h
```
python parser.py -h
usage: parser.py [-h] [-d DIR] [-s SPLIT] [-f {1,2}] [-i IMAGE_DIR]
                 [-md MODEL_DIR] [-mc MODEL_CHECKPT] [--batch_size BATCH_SIZE]
                 [-m {0,1,2,3}] [-o OUTPUT_DIR] [-ud {0,1}]

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
  -m {0,1,2,3}, --mode {0,1,2,3}
                        Specify the operation to perform.
                        0. Default mode which runs all the operations 
                        1. Simple image
                        2. Optical flow image
                        3. Depth image
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Specified the output location for the code.
  -ud {0,1}, --use_default {0,1}
                        Set to 1 to use default directories for all the models 
                        else set to 0 to specify all directories manually.


```
### Run as
```
python parser.py -d <input directory path> --output_dir <output directory path> --mode <any one of 0,1,2,3> --batch_size <number> -f < 2 if it is a directory of videos or images >
```
Incase the mode is specified to be 0, the model directories and checkpoints are not required as they are taken from the default values specified.
In other cases, model directory and checkpoint need to be specified.



### Visual Odometry

Using opencv, it utilises the output from FlowNet2.0 to generate the trajectory.

```
python odometry2d.py
```



### Future Scope

1. Add sparse scene flow using motion segmentation.
2. Fix deep visual odometry model.