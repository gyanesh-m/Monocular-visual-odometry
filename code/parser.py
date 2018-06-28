import argparse
import os

def main():
	parser = argparse.ArgumentParser(description="Generates the training data for specified flows.",
									 formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('-d', '--dir', type=str, help="""Video data directory path.\nThis is the path of the folder which contains subfolders""")
	parser.add_argument('-s', '--split', type=str, help='Location of Train/Test split file')
	parser.add_argument('-f', '--format-name', type=int,choices=[1,2],
						help="Specify the format number according to the following mapping-\n"+
						"1. EGTEA+ dataset format(Nested folders) \n2. Simple Image Folder",default=1)
	parser.add_argument("-i","--image-dir",type=str,help="Location of image directory",
						default=os.path.join(os.getcwd(),"./../data/train/"))
	parser.add_argument("-md","--model-dir",type=str,help="Location of the optical flow / depth model used.",
						default=os.path.join(os.getcwd(),"/flownet2-pytorch/utils/flownet2-pytorch"))
	parser.add_argument("-mc", "--model-checkpt", type=str, help="Location of optical flow / depth model checkpoint",
						default=os.path.join(os.getcwd(),"/flownet2-pytorch/utils/flownet2-pytorch/FlowNet2_checkpoint.pth.tar"))
	parser.add_argument("--batch_size",type=str,help="Mention the batch size for depth and optical flow.",default='8')
	parser.add_argument("--mode",type=int, choices=range(0,4),help="Specify the operation to perform.\n0. Default mode "+
						"which runs all the operations \n1. Simple image\n2. Optical flow image\n3. Depth image"
	"""\n""")
	args = parser.parse_args()
	return args
