from subprocess import call
import argparse
from merge_motion import run as merge_it
def run(FLAGS):
	base_dir=FLAGS.base_dir
	dest="OchsBroxMalik8_{}_{}_0000060.00"
	foldername=FLAGS.folder_name
	s="./MoSeg filestruct.cfg "+foldername+" {} {} 8"
	gpus="./dens100gpu fileDense.cfg mov2/mov2.ppm \
	             "+dest+"/"+foldername+"/Tracks{}.dat -1 \
	             "+dest+"/"+foldername+"/DenseSegmentation"
	multi_cut="./motionseg_release /media/DataDriveA/Datasets/gyanesh/frames/{}/{}.bmf {} {} 8 0.5"
	typeof={0:'cpu-multi cut',1:'cpu-moseg-longterm',2:'gpu'}
	def f(inp):
		return str(inp)
	gpu=FLAGS.operation
	start,stop,splits=FLAGS.start,FLAGS.stop,FLAGS.splits
	ans=''
	count=1
	steps=int((stop-start+1)/splits)
	list_dirs=[]
	for i in range(start,stop,steps):
		# print(i)
		if(count<splits):
			if(gpu==2):
				ans+=gpus.format(f(i).zfill(4),f(steps).zfill(4),f(steps),f(i).zfill(4),f(steps).zfill(4))
				list_dirs.append(dest.format(f(i).zfill(4),f(steps).zfill(4)))
			elif(gpu==0):
				ans+=multi_cut.format(foldername,foldername,i,steps)
			else:
				ans+=s.format(i,steps)
				list_dirs.append(dest.format(f(i).zfill(4),f(steps).zfill(4)))
			print(i,steps)
			ans+=' & '
		else:
			print('greater or equal')
			print(i,steps)
			if(gpu==2):
				ans+=gpus.format(f(i).zfill(4),f(stop-i+1).zfill(4),f(stop-i+1).zfill(4),f(i).zfill(4),f(stop-i+1).zfill(4))
				list_dirs.append(dest.format(f(i).zfill(4),f(stop-i+1)))	
			elif(gpu==0):
				ans+=multi_cut.format(foldername,foldername,i,stop-i+1)
			else:
				ans+=s.format(i,stop-i+1)
				list_dirs.append(dest.format(f(i).zfill(4),f(steps).zfill(4)))
			break
		count+=1
	with open(foldername+"_"+str(typeof[gpu])+"splits"+f(splits)+"-work.sh","w")	as f:
		f.write(ans)
	with open("dir-"+str(typeof[gpu])+".txt","w") as f:
		for folders in list_dirs:
			f.write(base_dir+folders+"/"+foldername+"/\n")
	print("Running-"+str(gpu)+"split-work.sh")
	call(["bash","./"+str(gpu)+"split-work.sh"])

def main():
	parser = argparse.ArgumentParser(description="Generates the sparse segmenation outputs.")
	parser.add_argument('-o','--base_dir',help="Specify the output direcotry",type=str,default="/media/DataDriveA/Datasets/gyanesh/frames/results/")
	parser.add_argument('-f','--folder_name',type=str,help="Specify the folder name")
	parser.add_argument('-m','--operation',choices=range(0,3),type=int,help="Specify the mode of operation\n"+"0.cpu-multicut\n1.cpu-moseg-longterm\n2.gpu")
	parser.add_argument('--start',type=int,help="Specify the first frame number")
	parser.add_argument('--stop',type=int,help="Specify the last frame number")
	parser.add_argument('--splits',type=int,help="Specify the times to split the data for parallel processing.")
	FLAGS = parser.parse_args()
	print("Generating segmenation for the splitted data.")
	run(FLAGS)
	print("Merging the splitted data.")
	merge_it(FLAGS)

if __name__ == '__main__':
	main()