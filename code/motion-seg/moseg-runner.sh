#python parser.py -i /home/gyanesh/output/Single/teleout/ -o /home/gyanesh/output/ -f 2 -m 2
#python parser.py -d /home/gyanesh/tiny/ -o /home/gyanesh/trial/ -f 2 -m 0


#for i in /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/00/image_2/*.png;do convert $i -resize 30% $i;done
#python parser.py -i /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/00/image_2/ -o /media/DataDriveA/Datasets/gyanesh/vo_test/00/ -f 2 -m 3;
#for i in /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/03/image_2/*.png;do convert $i -resize 30% $i;done
#python parser.py -i /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/03/image_2/ -o /media/DataDriveA/Datasets/gyanesh/vo_test/03/ -f 2 -m 3;
#for i in /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/05/image_2/*.png;do convert $i -resize 30% $i;done
#python parser.py -i /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/05/image_2/ -o /media/DataDriveA/Datasets/gyanesh/vo_test/05/ -f 2 -m 3;
#exit;
#python parser.py -i /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/06/image_2/ -o /media/DataDriveA/Datasets/gyanesh/vo_test/06/ -f 2 -m 3;
#python parser.py -i /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/07/image_2/ -o /media/DataDriveA/Datasets/gyanesh/vo_test/07/ -f 2 -m 3;
#python parser.py -i /media/DataDriveA/Datasets/gyanesh/poses/dataset/sequences/08/image_2/ -o /media/DataDriveA/Datasets/gyanesh/vo_test/08/ -f 2 -m 3;
echo "Testing tump-dutch video."
python parser.py -d ./../new_video/ -o /media/DataDriveA/Datasets/gyanesh/ -f 2 -m 0;