from glob import glob
import numpy as np
import cv2
import timeit
from math import fabs
seq='03'

DATA_DIR="/media/DataDriveA/Datasets/gyanesh/output-vo-mid/"
IMG_DIR=DATA_DIR+'/{}/Single/'.format(seq)
FLOW_DIR=DATA_DIR+'/{}/OpticalFlow/flo/'.format(seq)
POSE_DIR='/media/DataDriveA/Datasets/gyanesh/poses/dataset/poses/'
# SAVE_DIR=DATA_DIR+'/{}'.seq

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
      
def featureTracking2(img_1, img_2, p1):

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2

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
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

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


def featureTracking(image_ref,image_cur,keypoints,flow_lower_bound=5,index_num=0):
    #computes the features from the optical flow obtained from flownet 2.0  
    gray1=image_ref
    gray2=image_cur
    flow_farne=cv2.calcOpticalFlowFarneback(gray1,gray2,flow=None,pyr_scale=0.5, levels=3, winsize=12,
                                        iterations=3,
                                        poly_n=5, poly_sigma=1.2,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    flow_net=readFlow(FLOW_DIR+"/{0:06d}.flo".format(index_num))
    row,col=keypoints.shape
    previous_ref=[]
    current_ref=[]
    per=0
    print("keypoints",keypoints.shape)
    print("keypoints",keypoints[0].max(),keypoints[1].max()) 
    step = 50
    h,w=flow_net.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow_net[y,x].T
    previous_ref = np.array([x,y]).reshape(-1,2)
    current_ref = np.array([x+fx,y+fy]).reshape(-1,2)
    print("len",len(previous_ref))
    print("len",len(current_ref))
    prev=np.array(previous_ref,dtype=np.float32).reshape(-1,2)
    curr=np.array(current_ref,dtype=np.float32).reshape(-1,2)
    return prev,curr
def featureDetection():
    thresh = dict(threshold=25, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast

def getTruePose():
    file = POSE_DIR+'{}.txt'.format(seq)
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def getImages(i):
    print("dir",IMG_DIR+'/{0:06d}.png'.format(i))
    return cv2.imread(IMG_DIR+'/{0:06d}.png'.format(i), 0)

def getK():
    k="7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 4.538225000000e+01 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 -1.130887000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.779761000000e-03"
    arr = [float(i) for i in k.split()]
    return np.array(arr)

#initialization
ground_truth =getTruePose()

img_1 = getImages(0)
img_2 = getImages(1)
# print(img_1)

if len(img_1) == 3:
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
else:
    gray_1 = img_1
    gray_2 = img_2

#find the detector
detector = featureDetection()
kp1      = detector.detect(img_1)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
p1, p2   = featureTracking(gray_1, gray_2, p1)

#Camera parameters
fc = 718.8560
pp = (607.1928, 185.2157)
K  = getK()

E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0); 
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp);

#initialize some parameters
MAX_FRAME     = len(glob(IMG_DIR+'/*.png'))
MIN_NUM_FEAT  = 200

preFeature = p2
preImage   = gray_2

R_f = R
t_f = t

start = timeit.default_timer()

traj = np.zeros((1500, 1500, 3), dtype=np.uint8)

maxError = 0

#play image sequences
skipped=[]


for numFrame in range(2, MAX_FRAME):
    not_skip=True
    print(numFrame)

    if (len(preFeature) < MIN_NUM_FEAT):
        feature   = detector.detect(preImage)
        preFeature = np.array([ele.pt for ele in feature],dtype='float32')
    curImage_c = getImages(numFrame)

    if len(curImage_c) == 3:
          curImage = cv2.cvtColor(currImage_c, cv2.COLOR_BGR2GRAY)
    else:
          curImage = curImage_c
    
    kp1 = detector.detect(curImage)
    preFeature, curFeature = featureTracking(preImage, curImage, preFeature,index_num=numFrame-2)
    start_bound=10  
    if(len(preFeature)>10 and len(curFeature)>10):
        E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC,0.999,1.0); 
        not_skip=True
    else:
        not_skip=False
        pass
    if(not_skip):
        _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp = pp);
    
    truth_x, truth_y, truth_z, absolute_scale = getAbsoluteScale(ground_truth, numFrame)
    if not_skip and absolute_scale > 0.1:  
        t_f = t_f + absolute_scale*R_f.dot(t)
        R_f = R.dot(R_f)
    
    preImage = curImage
    preFeature = curFeature
    

    ####Visualization of the result
    if(not_skip):
        draw_x, draw_y = int(t_f[0]) + 300, int(t_f[2]) + 100;
    draw_tx, draw_ty = int(truth_x) + 300, int(truth_z) + 100
    if(not_skip):
        curError = np.sqrt((t_f[0]-truth_x)**2 + (t_f[1]-truth_y)**2 + (t_f[2]-truth_z)**2)
        print('Current Error: ', curError)
        if (curError > maxError):
            maxError = curError
    if(not_skip):
        cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2);
    cv2.circle(traj, (draw_tx, draw_ty) ,1, (255,0,0), 2);

    cv2.rectangle(traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED);
    text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(t_f[0]), float(t_f[1]), float(t_f[2]));
    cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8);
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
          break

print('Maximum Error: ', maxError)
cv2.imwrite('map-{}.png'.format(seq), traj);
stop = timeit.default_timer()
print(stop - start)
print(skipped)
cv2.destroyAllWindows()
