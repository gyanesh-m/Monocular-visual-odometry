#!/usr/bin/env python
import os
import argparse
from subprocess import call
from glob import glob
import shutil
OPTICAL_FLOW_MODEL='FlowNet2'
DEPTH_MODEL="geonet"

class Preprocess(object):
    """Extracts images from videos for the two formats. Thereafter, it generates images for all the four phases.

    """
    def __init__(self, data_dir, split_dir, format_name, img_dir, model_checkpt=None, model_dir=None,
                 batch=1):
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.format = format_name
        self.img_dir=img_dir
        self.model_checkpt=model_checkpt
        self.model_dir=model_dir
        self.batch_size=batch
    
    def extract_images(self):
        """Returns the sequence of videos in the train/test split file from EGTEA+ dataset
        """
        #_data contains training and test video file strings from the split file.
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
                print("Doing for file "+file_name)
                call(['ffmpeg', '-i', os.path.join(self.data_dir, file_name, ' '.join(file_line.split(" ")[:-3])+'.mp4')
                      , os.path.join(single_frame_loc, labels[key][file_line[:-4]], file_name[:22]+'%06d.png')])
    
    def generate_optical_flow_files(self):
        """Generates the flow files for the images in the dir specified(default Single dir)
        """
        for folder in os.listdir(os.path.join(self.img_dir,"Single")):
            print(folder,os.path.isdir(os.path.join(self.img_dir,"Single",folder)))
            if(os.path.isdir(os.path.join(self.img_dir,"Single",folder))):
                call(['python', os.path.join(self.model_dir,'main.py'),'--inference', '--model',
                OPTICAL_FLOW_MODEL, '--save_flow', '--inference_dataset', 'ImagesFromFolder', 
                '--inference_dataset_root', os.path.join(self.img_dir,'Single',folder), '--resume', self.model_checkpt,
                '--batch_size',self.batch_size])    
                # Extracts the directories in order of least modified first, i.e., last modified
                # directory is the last element in the list
                epoch_dir=os.listdir(os.path.join(self.model_dir,'work','inference'))
                flow_dir = os.path.join(self.model_dir, 'work', 'inference', epoch_dir[0])
                try:
                    os.makedirs(os.path.join(self.img_dir,'OpticalFlow',folder))
                except OSError as e:
                    pass
                #Generate the flow images as png and relocate to the train folder
                epoch_dir=os.listdir(os.path.join(self.model_dir,'work','inference'))
                flow_dir = os.path.join(self.model_dir, 'work', 'inference', epoch_dir[0])
                for flow_files in glob(flow_dir):
                    print(flow_dir)
                    call(['./color_flow',flow_files,flow_files[:-3]+'.png'])
                    os.rename(flow_files[:-3]+'.png',os.path.join(self.img_dir,'OpticalFlow',folder))
                shutil.rmtree(flow_dir)

    def generate_mono_depth(self):
        """Generates monocular depth image for the images in the data directory."""
        if(self.format==1):
        #For EGTEA data format.
            for folder in os.listdir(os.path.join(self.img_dir,"Single")):
                if(os.path.isdir(os.path.join(self.img_dir,"Single",folder))):
                    try:
                        os.makedirs(os.path.join(self.img_dir,'Depth',folder))
                    except OSError as e:
                        pass
                    cmd=['python',DEPTH_MODEL+'_main.py','--mode','test_depth','--dataset_dir',
                    os.path.join(self.img_dir,'Single',folder),'--init_ckpt_file',self.model_checkpt,
                    '--batch_size',self.batch_size,"--output_dir",os.path.join(self.img_dir,'Depth',folder)]
                    call(cmd)
        if(self.format==2):
        #For any general Image folder format
            if(os.path.isdir(os.path.join(self.img_dir))):
                try:
                    out_dir=os.path.join(self.img_dir,"../Depth")
                    os.makedirs(out_dir)
                except OSError as e:
                    pass
                cmd=['python',os.path.join(self.model_dir,'geonet_main.py'),'--mode','test_depth','--dataset_dir',
                    os.path.join(self.img_dir),'--init_ckpt_file',self.model_checkpt,
                    '--batch_size',self.batch_size,"--output_dir",out_dir]
                call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Generates the training data for specified flows.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, help="""Video data directory path.\nThis is the path of the folder which contains subfolders""")
    parser.add_argument('-s', '--split', type=str, help='Location of Train/Test split file')
    parser.add_argument('-f', '--format-name', type=int,
                        help="Specify the format number according to the following mapping-\n"+
                        "1. EGTEA+ dataset format \n2. Simple Image Folder",default=1)
    parser.add_argument("-i","--image-dir",type=str,help="Location of image directory",
                        default=os.path.join(os.getcwd(),"./../data/train/"))
    parser.add_argument("-md","--model-dir",type=str,help="Location of the optical flow model used.",
                        default=os.path.join(os.getcwd(),"/flownet2-pytorch/utils/flownet2-pytorch"))
    parser.add_argument("-mc", "--model-checkpt", type=str, help="Location of optical flow model checkpoint",
                        default=os.path.join(os.getcwd(),"/flownet2-pytorch/utils/flownet2-pytorch/FlowNet2_checkpoint.pth.tar"))
    parser.add_argument("--batch_size",type=str,help="Mention the batch size for depth and optical flow.",default='8')
    parser.add_argument("--mode",type=int, choices=range(0,4),help="Specify the operation to perform.\n0. Default mode "+
                        "which runs all the operations \n1. Simple image\n2. Optical flow image\n3. Depth image"
    """\n""")
    args = parser.parse_args()
    data_dir = args.dir
    split_dir = args.split
    format_name = args.format_name
    image_dir=args.image_dir
    model_dir=args.model_dir
    model_checkpt=args.model_checkpt
    batch=args.batch_size
    mode=args.mode
    process_obj = Preprocess(data_dir, split_dir, format_name, image_dir, model_checkpt, model_dir, batch)
    if(mode==1 or mode==0):
        process_obj.extract_images()
    elif(mode==2 or mode==0):
        process_obj.generate_optical_flow_files()
    elif(mode==3 or mode==0):
        process_obj.generate_mono_depth()

if __name__ == '__main__':
    main()