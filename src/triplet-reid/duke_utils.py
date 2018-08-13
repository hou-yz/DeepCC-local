import cv2
import math
import numpy as np
import csv
import os
import scipy.io as sio
import h5py
import tensorflow as tf

class DukeVideoReader:
# Use
# reader = DukeVideoReader('g:/dukemtmc/')
# camera = 2
# frame = 360720
# img = reader.getFrame(camera, frame)

    def __init__(self, dataset_path):
        self.NumCameras = 8
        self.NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
        self.PartMaxFrame = 38370
        self.MaxPart = [9, 9, 9, 9, 9, 8, 8, 9]
        self.PartFrames = []
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 14250])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 15390])
        self.PartFrames.append([38400, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 10020])
        self.PartFrames.append([38670, 38670, 38670, 38670, 38670, 38700, 38670, 38670, 38670, 26790])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 21060])
        self.PartFrames.append([38400, 38370, 38370, 38370, 38400, 38400, 38370, 38370, 37350, 0])
        self.PartFrames.append([38790, 38640, 38460, 38610, 38760, 38760, 38790, 38490, 28380, 0])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 7890])
        self.DatasetPath = dataset_path
        self.CurrentCamera = 1
        self.CurrentPart = 0
        self.PrevCamera = 1
        self.PrevFrame = -1
        self.PrevPart = 0
        self.Video = cv2.VideoCapture('{:s}videos/camera{:d}/{:05d}.MTS'.format(self.DatasetPath, self.CurrentCamera, self.CurrentPart), cv2.CAP_FFMPEG)

    def getFrame(self, iCam, iFrame):
        # iFrame should be 1-indexed
        assert iFrame > 0 and iFrame <= self.NumFrames[iCam-1], 'Frame out of range'
        #print('Frame: {0}'.format(iFrame))
        # Cam 4 77311
        #Coompute current frame and in which video part the frame belongs
        ksum = 0
        for k in range(10):
            ksumprev = ksum
            ksum += self.PartFrames[iCam-1][k]
            if iFrame <= ksum:
                currentFrame = iFrame - 1 - ksumprev
                iPart = k
                break
        # Update VideoCapture object if we are reading from a different camera or video part
        if iPart != self.CurrentPart or iCam != self.CurrentCamera:
            self.CurrentCamera = iCam
            self.CurrentPart = iPart
            self.PrevFrame = -1
            self.Video = cv2.VideoCapture('{:s}videos/camera{:d}/{:05d}.MTS'.format(self.DatasetPath, self.CurrentCamera, self.CurrentPart), cv2.CAP_FFMPEG)
        # Update time only if reading non-consecutive frames
        if not currentFrame == self.PrevFrame + 1:
            
            #if iCam == self.PrevCamera and iPart == self.PrevPart and currentFrame - self.PrevFrame < 30:
            #    # Skip consecutive images if less than 30 frames difference
            #    back_frame = max(self.PrevFrame, 0)
            #else:
            #    # Seek, but make sure a keyframe is read before decoding
            back_frame = max(currentFrame - 31, 0) # Keyframes every 30 frames
            #print('back_frame set to: {0}'.format(back_frame))
            self.Video.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
            if not self.Video.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
                print('Warning: OpenCV has failed to set back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(back_frame, self.Video.get(cv2.CAP_PROP_POS_FRAMES), currentFrame))
          
            back_frame = self.Video.get(cv2.CAP_PROP_POS_FRAMES)
            #print('back_frame is: {0}'.format(back_frame))
            while back_frame < currentFrame:
                self.Video.read()
                back_frame += 1
        #print('currentFrame: {0}'.format(currentFrame))
        #print('current position: {0}'.format(self.Video.get(cv2.CAP_PROP_POS_FRAMES)))
        assert self.Video.get(cv2.CAP_PROP_POS_FRAMES) == currentFrame, 'Frame position error'
        result, img = self.Video.read()
        if result is False:
            print('-Could not read frame, trying again')
            back_frame = max(currentFrame - 61, 0)
            self.Video.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
            if not self.Video.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
                print('-Warning: OpenCV has failed to set back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(back_frame, self.Video.get(cv2.CAP_PROP_POS_FRAMES), currentFrame))
            back_frame = self.Video.get(cv2.CAP_PROP_POS_FRAMES)
            #print('-back_frame is: {0}'.format(back_frame))
            while back_frame < currentFrame:
                self.Video.read()
                back_frame += 1
            result, img = self.Video.read()


        img = img[:, :, ::-1]  # bgr to rgb
        # Update
        self.PrevFrame = currentFrame
        self.PrevCamera = iCam
        self.PrevPart = iPart
        return img

def pose2bb(pose):

    renderThreshold = 0.05
    ref_pose = np.array([[0.,   0.], #nose
       [0.,   23.], # neck
       [28.,   23.], # rshoulder
       [39.,   66.], #relbow
       [45.,  108.], #rwrist
       [-28.,   23.], # lshoulder
       [-39.,   66.], #lelbow
       [-45.,  108.], #lwrist
       [20., 106.], #rhip
       [20.,  169.], #rknee
       [20.,  231.], #rankle
       [-20.,  106.], #lhip
       [-20.,  169.], #lknee
       [-20.,  231.], #lankle
       [5.,   -7.], #reye
       [11.,   -8.], #rear
       [-5.,  -7.], #leye
       [-11., -8.], #lear
       ])
       
   
    # Template bounding box   
    ref_bb = np.array([[-50., -15.], #left top
                [50., 240.]])  # right bottom
            
    pose = np.reshape(pose,(18,3))
    valid = np.logical_and(np.logical_and(pose[:,0]!=0,pose[:,1]!=0), pose[:,2] >= renderThreshold)

    if np.sum(valid) < 2:
        bb = np.array([0, 0, 0, 0])
        print('got an invalid box')
        print(pose)
        return bb

    points_det = pose[valid,0:2]
    points_reference = ref_pose[valid,:]
    
    # 1a) Compute minimum enclosing rectangle

    base_left = min(points_det[:,0])
    base_top = min(points_det[:,1])
    base_right = max(points_det[:,0])
    base_bottom = max(points_det[:,1])

    # 1b) Fit pose to template
    # Find transformation parameters
    M = points_det.shape[0]
    B = points_det.flatten('F')
    A = np.vstack((np.column_stack((points_reference[:,0], np.zeros((M)), np.ones((M)),  np.zeros((M)))),
         np.column_stack((np.zeros((M)),  points_reference[:,1], np.zeros((M)),  np.ones((M))) )))
    
    
    params = np.linalg.lstsq(A,B)
    params = params[0]
    M = 2
    A2 = np.vstack((  np.column_stack( (ref_bb[:,0], np.zeros((M)), np.ones((M)),  np.zeros((M)))),
         np.column_stack( (np.zeros((M)),  ref_bb[:,1], np.zeros((M)),  np.ones((M)))) ))

    result = np.matmul(A2,params)

    fit_left = min(result[0:2])
    fit_top = min(result[2:4])
    fit_right = max(result[0:2])
    fit_bottom = max(result[2:4])

    # 2. Fuse bounding boxes
    left = min(base_left,fit_left)
    top = min(base_top,fit_top)
    right = max(base_right,fit_right)
    bottom = max(base_bottom,fit_bottom)

    left = left*1920
    top = top*1080
    right = right*1920
    bottom = bottom*1080

    height = bottom - top + 1
    width = right - left + 1

    bb = np.array([left, top, width, height])
    return bb

def scale_bb( bb, pose, scalingFactor ):
    # Scales bounding box by scaling factor
    newbb = np.zeros(bb.shape)
    newbb[0:2] = bb[0:2] - 0.5*(scalingFactor-1) * bb[2:4]
    newbb[2:4] = bb[2:4] * scalingFactor

    # X, Y, strength
    newpose = np.reshape(pose,(18,3))
    # Scale to original bounding box
    newpose[:,0] = (newpose[:,0] - bb[0]/1920.0) / (bb[2]/1920.0)
    newpose[:,1] = (newpose[:,1] - bb[1]/1080.0) / (bb[3]/1080.0)

    # Scale to stretched bounding box
    newpose[:,0] = (newpose[:,0] + 0.5*(scalingFactor-1))/scalingFactor
    newpose[:,1] = (newpose[:,1] + 0.5*(scalingFactor-1))/scalingFactor
    # Return in the original format
    newpose[newpose[:,2]==0,0:2] = 0
    newpose = np.ravel(newpose)

    return newbb, newpose

def feet_position(boxes):
    
    x = boxes[0] + 0.5*boxes[2];
    y = boxes[1] + boxes[3];
    feet = np.array([x, y]);
    return feet

def get_bb(img, bb):
    bb = np.round(bb)
    
    left = np.maximum(0,bb[0]).astype('int')
    right = np.minimum(1920-1,bb[0]+bb[2]).astype('int')
    top = np.maximum(0,bb[1]).astype('int')
    bottom = np.minimum(1080-1,bb[1]+bb[3]).astype('int')
    if left == right or top == bottom:
    	return np.zeros((256,128,3))
    snapshot = img[top:bottom,left:right,:]
    return snapshot

def convert_img(img):
    img = img.astype('float')
    img = img / 255.0
    img = img - 0.5
    return img

def detections_generator(base_path, detections, height, width):

    reader = DukeVideoReader(base_path)

    for ind in range(detections.shape[0]):
        print('reading detection {0}/{1}'.format(ind+1,detections.shape[0]))
        camera = int(detections[ind][0])
        frame  = int(detections[ind][1])
        box    = detections[ind][2:6]
        img = reader.getFrame(camera, frame)
     
        if box[2] < 20 or box[3] < 20:
            snapshot = np.zeros((height,width,3))
        else:
            snapshot = get_bb(img, box)
            snapshot = cv2.resize(snapshot,(width, height))  

        yield snapshot

def detections_generator_from_openpose(iCam, base_path, detections_path):

    reader = DukeVideoReader(base_path)
    #for iCam in range(1,9):
    prev_frame = -1
    pose_file = os.path.join(detections_path,'camera{0}_openpose.mat'.format(iCam))

    with h5py.File(pose_file, 'r') as f:
        detections = np.transpose(np.array(f['detections']))

    for ind in range(detections.shape[0]):

        iFrame = detections[ind,1].astype('int')
        if iFrame != prev_frame:
            img = reader.getFrame(iCam,iFrame)
            prev_frame = iFrame

        pose = detections[ind,2:]
        bb = pose2bb(pose)
        newbb, newpose = scale_bb(bb,pose,1.25)

        if newbb[2] < 20 or newbb[3] < 20:
            snapshot = np.zeros((256,128,3))
        else:
            snapshot = get_bb(img, newbb)
            snapshot = cv2.resize(snapshot,(128, 256))  

        yield snapshot
        
        