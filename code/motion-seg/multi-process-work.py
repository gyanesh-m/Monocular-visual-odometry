from subprocess import call
import argparse
from merge_motion import run as merge_it
from utils import *
import time

##TODO
# Currently dense segmenation works for long term.
# Need to update it to work with multi cut segmentation.

def run(FLAGS):
	data_dir=FLAGS.data_dir+"/results/"
	dest="OchsBroxMalik8_{}_{}_0000060.00"
	foldername=FLAGS.folder_name
	s="./scripts/MoSeg cpu.cfg "+foldername+" {} {} 8"
	gpus="./scripts/dens100gpu gpu.cfg "+foldername+"/"+foldername+".ppm \
	             "+dest+"/"+foldername+"/Tracks{}.dat -1 \
	             "+dest+"/"+foldername+"/DenseSegmentation"
	multi_cut="./scripts/motionseg_release "+FLAGS.data_dir+"/{}/{}.bmf {} {} 8 0.5"
	typeof={0:'cpu-multi-cut',1:'cpu-moseg-longterm',2:'gpu'}
	def f(inp,fill=1):
		if(fill):
			return str(inp).zfill(4)
		else:
			return str(inp)
	gpu=FLAGS.operation
	start,stop,splits=FLAGS.start,FLAGS.stop,FLAGS.splits
	ans=''
	count=1
	steps=int((stop-start+1)/splits)
	list_dirs=[]
	for i in range(start,stop,steps):
		if(count<splits):
			print(i,steps)
			if(gpu==2):
				ans+=gpus.format(f(i),f(steps),f(steps,0),f(i),f(steps))
				list_dirs.append(dest.format(f(i),f(steps)))
			elif(gpu==0):
				ans+=multi_cut.format(foldername,foldername,i,steps)
			else:
				ans+=s.format(i,steps)
				list_dirs.append(dest.format(f(i),f(steps)))
			print(i,steps)
			ans+=' & '
		else:
			print(i,stop-i)
			if(gpu==2):
				ans+=gpus.format(f(i),f(stop-i),f(stop-i,0),f(i),f(stop-i))
				list_dirs.append(dest.format(f(i),f(stop-i)))	
			elif(gpu==0):
				ans+=multi_cut.format(foldername,foldername,i,stop-i+1)
			else:
				ans+=s.format(i,stop-i)
				list_dirs.append(dest.format(f(i),f(stop-i)))
			break
		count+=1
	try:
		os.makedirs("./scripts/")
	except Exception as e:
		print(e)
	sh_name=foldername+"_"+str(typeof[gpu])+"_splits_"+f(splits)+"-work.sh"
	print("./scripts/"+sh_name)
	with open("./scripts/"+sh_name,"w")	as f:
		f.write(ans)
	try:
		os.makedirs("./dir_log/")
	except Exception as e:
		print(e)
	with open("./dir_log/dir-"+str(typeof[gpu])+".txt","w") as f:
		for folders in list_dirs:
			f.write(data_dir+"/"+folders+"/"+foldername+"/\n")
	
	bash_path="./scripts/"+sh_name
	print("Running "+bash_path)
	call(["bash",bash_path])

def main():
	parser = argparse.ArgumentParser(description="Generates the sparse segmenation outputs.")
	parser.add_argument('-d','--data_dir',help="Parent folder of folders which contains sequences of frames.",type=str,default="/media/DataDriveA/Datasets/gyanesh/frames/")
	parser.add_argument("-o","--output_dir",help="Specify the output directory for segmented data",type=str, default="/datadir/../Multicut/foldername/")
	parser.add_argument('-f','--folder_name',type=str,help="Specify the folder name")
	parser.add_argument('-m','--operation',choices=range(0,3),type=int,help="Specify the mode of operation \n0."+"cpu-multicut\n1.cpu-moseg-longterm\n2.gpu",default=0)
	parser.add_argument('--start',type=int,help="Specify the first frame number\n"+"Defaults to 1.",default=1)
	parser.add_argument('--stop',type=int,help="Specify the last frame number.\n"+"Defaults to total number of frames present.",default=1111)
	parser.add_argument('--splits',type=int,help="Specify the times to split the data for parallel processing. Doesn't work for mode 0.")
	FLAGS = parser.parse_args()
	if(FLAGS.operation==2):
		iscpu=0
	else:
		iscpu=1
	print("Updating config files.")
	if(FLAGS.output_dir=="/datadir/../Multicut/foldername/"):
		FLAGS.output_dir=FLAGS.data_dir+"/../"
	make_config(iscpu,FLAGS.data_dir)
	print("Creating bmf file for "+FLAGS.folder_name)

	folder_path=FLAGS.data_dir+"/"+FLAGS.folder_name+"/"
	make_bmf(folder_path)
	if(FLAGS.stop==1111):
		FLAGS.stop=len(glob(folder_path+"/*.ppm"))-1
	print(FLAGS.start)
	print(FLAGS.stop)

	if(FLAGS.operation==0 and FLAGS.splits!=1):
		print("Splitting data with multi cut is not supported yet.")
		print("Changing split back to 1.")
		time.sleep(3)
	print("Generating segmenation for the splitted data.")
	run(FLAGS)
	if(FLAGS.operation!=0):
		print("Merging the splitted data.")
		merge_it(FLAGS)
	if(FLAGS.operation==0):
		print("Moving multi cut outputs")
		move_files(folder_path,FLAGS.output_dir,FLAGS.folder_name)
if __name__ == '__main__':
	main()