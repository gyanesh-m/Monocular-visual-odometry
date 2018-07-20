from scipy.misc import *
import numpy as np

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code taken from:
    # https://github.com/swehrwein/pipi/blob/master/flow.py#L12
    #
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
def get_depth_flow_and_3dflow(depth_path,flow_path,shape):
    """
    Takes depth image path and optical flow image path as input,
    and returns the depth flow.
    Depth flow is computed as follows:
    D_Flow_t_t+1 = D_t+1[(x,y)+Flow_t_t+1(x,y)]-D_t
    
    """
    ht,wd=shape
    depth_t = imread(depth_path[0])
    depth_t1 = imread(depth_path[1])

    depth_t=imresize(depth_t,(ht,wd))
    depth_t1=imresize(depth_t1,(ht,wd))

    index_depth=np.indices(depth_t.shape)
    ########################
    # not sure whether to use .flo file or their visualised .png file.
    # currently it depends on what input is supplied.
    ########################
    if(flow_path.endswith(".flo")):
        flow=readFlow(flow_path)
    elif(flow_path.endswith(".png")or flow_path.endswith(".jpg")):
        flow_=imread(flow_path)
        flow=np.resize(flow_,(ht,wd,2))
    k=index_depth+flow.reshape(index_depth.shape)
    modified_index=np.ndarray.astype(index_depth+flow.reshape(index_depth.shape),int)
    bool_index=modified_index[:]>=ht
    modified_index[bool_index]=ht-1
    bool_index=modified_index[:]>=wd
    modified_index[bool_index]=wd-1
    row=modified_index[0]
    col=modified_index[1]
    r,c=depth_t.shape
    new_depth_t1=[]
    new_depth_t1=depth_t1[row,col]
    new_depth_t1=np.array(new_depth_t1).reshape(depth_t.shape)
    depth_flow=new_depth_t1-depth_t
    # Can do smoothing aswell using
    # gaussian(depth_flow,sigma=0.4).reshape(ht,wd))
    depth_flow=depth_flow.reshape(1,ht,wd)
    flow=np.array(flow).reshape(-1,ht,wd)
    threeD_flow=np.vstack((flow,depth_flow))
    # For now, it returns depth flow output only.
    # Can also return 3d flow but it isn't returned,
    # since it is computed by the concat operation inside keras.
    return depth_flow.reshape(ht,wd)