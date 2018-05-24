#!/usr/bin/env python
import os
import argparse
from subprocess import call


class Preprocess(object):
    """Extracts images from videos for the two formats. Thereafter, it generates images for all the four phases.

    """

    def __init__(self, data_dir, split_dir, format_name):
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.format = format_name
        if self.format == 1:#EGTEA +
            self.extract_images()

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
                      , os.path.join(single_frame_loc, labels[key][file_line[:-4]], file_name[:22]+'%04d.jpg')])
def main():
    parser = argparse.ArgumentParser(description="Generates the training data for all the four flows.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, help="""Video data directory path.\n
                        This is the path of the folder which contains subfolders""", required=True)
    parser.add_argument('-s', '--split', type=str, help='Location of Train/Test split file', required=True)
    parser.add_argument('-f', '--format-name', type=int,
                        help="""Specify the format number according to the following mapping-\n
                        1 EGTEA+ dataset \n
                        2 ELAN
                        """)
    args = parser.parse_args()
    data_dir = args.dir
    split_dir = args.split
    format_name = args.format_name
    process_obj = Preprocess(data_dir, split_dir, format_name)
if __name__ == '__main__':
    main()
