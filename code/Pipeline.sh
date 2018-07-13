: 'Extracts frames, optical flow and depth .
Here m specifies the operation to perform
0. run all
1. frames
2. Optical flow
3. Depth
Here f specifies the file format
1 . EGTEA data format.
2 . Any video data directory or any image directory.
For complete parameter info : python parser.py -h
'
python parser.py -d input_dir -o output_dir -f 2 -m 0 ;
cd motion-seg;

: '
Runs motion segmentation
Specify the mode of operation:
0.cpu-multicut 
1.cpu-moseg-longterm 
2.gpu
Splits increases the code processing speed by running each data split parallely.
Splits only works with mode 1 as of now.
For complete parameter info : python multi-process-work -h
'
python multi-process-work.py -d parent_dir_containing_image_folders -f folder_name -o output_dir -m operation --splits number;
