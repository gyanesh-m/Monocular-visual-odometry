import argparse
import os
from pre_processing import *


def main():
	parser = argparse.ArgumentParser(description="Generates the training data for specified flows.",
									 formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('-d', '--dir', type=str, help="""Video data directory path.\nThis is the path of the folder which contains subfolders""")
	parser.add_argument('-s', '--split', type=str, help='Location of Train/Test split file')
	parser.add_argument('-f', '--format-name', type=int,choices=[1,2],
						help="Specify the format number according to the following mapping-\n"+
						"1. EGTEA+ dataset format(Nested folders) \n2. Simple Image Folder",default=1)
	parser.add_argument("-i","--image-dir",type=str,help="Location of image directory",
						)
	parser.add_argument("-md","--model-dir",type=str,help="Location of the optical flow / depth model used.",
						default=os.path.join(os.getcwd(),"/flownet2-pytorch/utils/flownet2-pytorch"))
	parser.add_argument("-mc", "--model-checkpt", type=str, help="Location of optical flow / depth model checkpoint",
						default=os.path.join(os.getcwd(),"/flownet2-pytorch/utils/flownet2-pytorch/FlowNet2_checkpoint.pth.tar"))
	parser.add_argument("--batch_size",type=str,help="Mention the batch size for depth and optical flow.",default='8')
	parser.add_argument("-m","--mode",type=int, choices=range(0,4),help="Specify the operation to perform.\n0. Default mode "+
						"which runs all the operations \n1. Simple image\n2. Optical flow image\n3. Depth image"
	"""\n""")
	parser.add_argument("-o","--output_dir",type=str,help="Specified the output location for the code.")
	parser.add_argument("-ud","--use_default",default=1,type=int,choices=range(2),help="Set to 1 to use default directories for all the models \n"+
		"else set to 0 to specify all directories manually.")
	args = parser.parse_args()
	data_dir = args.dir
	split_dir = args.split
	format_name = args.format_name
	image_dir=args.image_dir
	model_dir=args.model_dir
	model_checkpt=args.model_checkpt
	batch=args.batch_size
	mode=args.mode
	output_dir=args.output_dir
	use_default=args.use_default
	process_obj = Preprocess(split_dir, format_name, image_dir, model_checkpt, model_dir, batch,output_dir, data_dir)
	if(mode==1 or mode==0):
		process_obj.extract_images()
		print("Outputs saved at "+str(process_obj.output_dir))
	if(mode==2 or mode==0):
		if(use_default==1):
			print("using default")
			process_obj.model_dir="./../utils/flownet2-pytorch/"
			process_obj.model_checkpt="/home/gyanesh/models/flownet2/FlowNet2_checkpoint.pth.tar"
		process_obj.generate_optical_flow()
		print("Outputs saved at "+str(process_obj.output_dir))
	if(mode==3 or mode==0):
		if(use_default==1):
			process_obj.model_dir="./../utils/MegaDepth/"
			process_obj.model_checkpt="./../utils/MegaDepth/checkpoints/"
		process_obj.generate_depth()
		print("Outputs saved at "+str(process_obj.output_dir))
if __name__ == '__main__':
	main()