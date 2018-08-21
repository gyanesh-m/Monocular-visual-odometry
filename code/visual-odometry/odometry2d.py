from glob import glob
import numpy as np
import cv2
import timeit
from math import fabs
import argparse
import os
import time

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code taken from:
    # https://github.com/swehrwein/pipi/blob/master/flow.py#L12
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print ('Magic number incorrect. Invalid .flo file')
            return None

        dims = np.fromfile(f, np.int32, count=2)
        w, h = dims[0], dims[1]
        data = np.fromfile(f, np.float32, count=2*w*h)
        return np.reshape(data, (h, w, 2))

def getAbsoluteScale(f, frame_id):
      x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
      x    , y    , z     = f[frame_id][3], f[frame_id][7], f[frame_id][11]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return x, y, z, scale

def plotAndSave(flow,save_name,hsv):
    print('flow',flow.shape)
    print('hsv',hsv.shape)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imsave(DATA_DIR+"/"+save_name+".png",rgb)
    cv2.imwrite('opticalfb.png',frame2)
    cv2.imwrite('opticalhsv.png',rgb)

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.array(np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1),dtype='int32')
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    feat_img = np.copy(img)
    cv2.polylines(feat_img, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(feat_img, (x1, y1), 1, (0, 255, 0), -1)
    return feat_img

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def featureTracking(image_ref,image_cur,keypoints,flow_files,flow_lower_bound=0.1,flow_index='000000.flo'):
    ''' This method is the fastest of all and gives reasonably good predictions.'''
    ''' No need to convert the image to grayscale since the image is already loaded in grayscale with 0 parameter in imread'''
    gray1=image_ref
    gray2=image_cur
  #         flow=cv2.calcOpticalFlowFarneback(gray1,gray2,0.5,3,12,3,5,1.2)
  # 0.5,5,150,60,7,1.5,OPTFLOW_FARNEBACK_GAUSSIAN

    '''
    keypoints are (y,x) cordinate pairs
    range is y->1:384, x->1:128
    '''
    flow=readFlow(flow_files+"/"+flow_index)
    X=np.array(keypoints[:,0],dtype='int32')
    Y=np.array(keypoints[:,1],dtype='int32')
    one = X
    two = Y
    r,c=flow.shape[:2]
    X%=c
    Y%=r
  #Discards points whose flow value is less than or equal to 0.1
    idex=np.abs(flow[Y,X])>0.1

    ''' idex gives the True/False values at index of points of X and Y based on the flow condition.
    True_x and True_y gives the actual index values.
    Final index is the set of indices of points which satisfied the flow condition.
    '''
    true_index=idex[:,0]*idex[:,1]
    final_index_x,final_index_y = X[true_index],Y[true_index]
    fx, fy = flow[final_index_y,final_index_x].T
    x=final_index_x
    y=final_index_y
    previous_ref=[]
    current_ref = []
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    for (x1,y1), (x2,y2) in lines:
        previous_ref.append([x1,y1])
        current_ref.append([x2,y2])
    prev=np.array(previous_ref,dtype=np.float32).reshape(-1,2)
    curr=np.array(current_ref,dtype=np.float32).reshape(-1,2)
    return prev,curr

def featureDetection():
  #Selects features using the FAST algorithm for corner detection.
    thresh = dict(threshold=25, nonmaxSuppression=True)
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast

def getTruePose(pose_file):
    return np.genfromtxt(pose_file, delimiter=' ',dtype=None)

def getImages(i,img_dir):
    print("dir",img_dir+'/{0:06d}.png'.format(i))
    return cv2.imread(img_dir+'/{0:06d}.png'.format(i))

def plotTrajectory(img_files,flow_files,pose_file,output_dir):
  #initialization
  MAX_FRAME     = len(glob(img_files+'/*.png'))
  MIN_NUM_FEAT  = 1500
  ground_truth =getTruePose(pose_file)
  # get first two images
  img_1 = getImages(0,img_files)
  img_2 = getImages(1,img_files)

  #Find the features using FAST feature detector
  detector = featureDetection()
  kp1      = detector.detect(img_1)
  p1       = np.array([ele.pt for ele in kp1],dtype='float32')
  p1, p2   = featureTracking(img_1, img_2, p1,flow_files,flow_index='000000.flo')

  #Camera intrinsic parameters, focal length and principal point
  fc = 718.8560/3
  pp = (607.1928/3, 185.2157/3)
    #Gets the essential matrix
  E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0)
  _, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp)
  preFeature = p2
  preImage   = img_2
  R_f = R
  t_f = t
  start = timeit.default_timer()
  #defines the grid where trajectory will be plotted.
  traj = np.zeros((1500, 1500, 3), dtype=np.uint8)
  maxError = 0
  skipped=[]
  for numFrame in range(2, MAX_FRAME):
    not_skip=True
    if (len(preFeature) < MIN_NUM_FEAT):     
        feature   = detector.detect(preImage)
        preFeature = np.array([ele.pt for ele in feature],dtype='int32')
      
    curImage_c = getImages(numFrame,img_files)
    # if(numFrame==2):
    curImage=np.copy(curImage_c)
    kp1 = detector.detect(curImage)
    preFeature, curFeature = featureTracking(preImage, curImage,preFeature,flow_files, flow_index=str((numFrame-1)).zfill(6)+'.flo')#,index_num=numFrame-2)
    start_bound=10
    if(len(preFeature)>10 and len(curFeature)>10):
      E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC,0.999,1.0)
      not_skip=True
    else:
      print("||skipping||"*10)
      print(numFrame)
      not_skip=False
      pass
    if(not_skip):
      _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp = pp)
    truth_x, truth_y, truth_z, absolute_scale = getAbsoluteScale(ground_truth, numFrame)
    if not_skip and absolute_scale > 0.1:
      t_f = t_f + absolute_scale*R_f.dot(t)
      R_f = R.dot(R_f)
    preImage = curImage
    preFeature = curFeature
    #Visualization of the result
    if(not_skip):
      draw_x, draw_y = -int(t_f[0]) + 500, int(t_f[2]) + 100
    draw_tx, draw_ty = -int(truth_x) + 500, int(truth_z) + 100
    if(not_skip):
      curError = np.sqrt((t_f[0]-truth_x)**2 + (t_f[1]-truth_y)**2 + (t_f[2]-truth_z)**2)
      print('Current Error: ', curError)
      if (curError > maxError):
        maxError = curError
    if(not_skip):
            #Plots predicted trajectory
      cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2)
    #Plots ground truth trajectory.
    cv2.circle(traj, (draw_tx, draw_ty) ,1, (255,0,0), 2)
    cv2.rectangle(traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED)
    #Displays the estimated coordinates.
    text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(t_f[0]), float(t_f[1]), float(t_f[2]))
    cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    #Draws the keypoints extracted for the current image which were tracked.
    cv2.drawKeypoints(curImage, kp1, curImage_c)
    #Plots the trajectory.
    cv2.imshow('Trajectory',traj)
    #Plots the features.
    cv2.imshow('Features', curImage_c)
    flow=readFlow(flow_files+"/"+str((numFrame-1)).zfill(6)+'.flo')
    img_hsv = draw_hsv(flow)
    img_flow = draw_flow(curImage,flow)
    #Plots the hsv values for the FlowNet2 output.
    cv2.imshow("HSV",img_hsv)
    #Plots the representation of the flow in the image for a grid of points, separated 16 units apart.
    cv2.imshow("FLOW",img_flow)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

  print('Maximum Error: ', maxError)
  fname='-'.join([str(i) for i in time.localtime()[:3]])
  fname =fname+"_"+ '-'.join([str(i) for i in time.localtime()[3:6]])
  print("saving Trajectory output to ")
  print(os.path.join(output_dir,'map-{}.png'.format(fname)))
  cv2.imwrite(os.path.join(output_dir,'map-{}.png'.format(fname)), traj)
  stop = timeit.default_timer()
  print("Time taken :"+str(stop - start)+'s')
  print("Skipped frames:",skipped)
  cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(description="Computes the trajectory, using output from FlowNet2.",
                   formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--img',type=str,help="Location of input image sequences.")
  parser.add_argument('--flo',type=str,help="Location of optical flow files.")
  parser.add_argument('--pose',type=str,help="Location of pose file.")
  parser.add_argument('--output',type=str,help="Location of output directory")
  args = parser.parse_args()
  img_files = os.path.join(args.img)
  flow_files = os.path.join(args.flo)
  pose_file = os.path.join(args.pose)
  output_dir = os.path.join(args.output)
  plotTrajectory(img_files,flow_files,pose_file,output_dir)

if __name__ == '__main__':
    main()