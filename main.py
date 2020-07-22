"""
Type following in commandline to execute
python main.py  -fd "Models\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -ge "Models\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002" -hpd "Models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001" -lmd "Models\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009" -i "bin\demo.mp4" -d CPU -pt 0.6 -m_prec "fast" -m_speed "high"

"""
import os
import sys
import time
import socket
import json
import cv2
import logging
import sys
import numpy as np

from argparse import ArgumentParser
from src.input_feeder import InputFeeder


from random import randint
from inference import Network

from src.face_detection import face_det_Model
from src.facial_landmarks_detection import Facial_Landmarks_Detection_Model
from src.head_pose_estimation import Head_pose_Model
from src.gaze_estimation import Gaze_est_Model 
from src.mouse_controller import MouseController

#import _thread



def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fdmodel", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-hpd", "--hpdmodel", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-lmd", "--lmdmodel", required=True, type=str,
                        help="Path to a facial landmarks xml file with a trained model.")
    parser.add_argument("-ge", "--gemodel", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path video file or CAM to use camera")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    #parser.add_argument("-v","--video",default=False,
     #                   help="Don't show video window",action="store_true")

    parser.add_argument("-m_prec", "--mouse_precision", type=str, default='high',
                        help="(Optional) Specify mouse precision (how much the mouse moves): 'high', 'medium', 'low'."
                             "Default is high.")
    parser.add_argument("-m_speed", "--mouse_speed", type=str, default='immediate',
                        help="(Optional) Specify mouse speed (how many secs before it moves): 'immediate'(0s), 'fast'(0.1s),"
                             "'medium'(0.5s) and 'slow'(1s). Default is immediate.")
    

    return parser


def infer_on_stream(args):
    fmodel = args.fdmodel
    flmodel = args.lmdmodel
    hmodel = args.hpdmodel
    gmodel = args.gemodel
    device = args.device
    input=args.input
    threshold = args.prob_threshold
    #initializing models
    face_model = face_det_Model(fmodel, device, threshold)
    facial_landmarks = Facial_Landmarks_Detection_Model(flmodel, device, threshold)
    head_pose_est = Head_pose_Model(hmodel,device, threshold)
    gaze_est = Gaze_est_Model(gmodel, device, threshold)

    # Loading models

    face_model.load_model()
    facial_landmarks.load_model()
    head_pose_est.load_model()
    gaze_est.load_model()
    

    if input != "cam":
        input_type = 'video'
    else:
        input_type = 'cam'

    counter = 0
    
        
    try:
        feed=InputFeeder(input_type = 'cam', input_file = input)
        feed.load_data()
        for batch in feed.next_batch():
            if batch is None:
                
                exit()
            counter += 1
            frame,face_crop = face_model.predict(batch)
            limg,rimg = facial_landmarks.predict(face_crop)
            angles,frame = head_pose_est.predict(face_crop,detections,frame)
            x,y = gaze_est.predict(limg,rimg,angles)
            
            if eye_detect == "eye_detect":
                cv2.imshow('frame', face_crop)
            elif head == "head_pose":
                cv2.imshow('frame',frame) 
            else:
                cv2.imshow('frame',frame)
                        
            if vtype != 'video':
                t=1
            else:
                t=500
                
            if cv2.waitKey(t) & 0xFF == ord('q'):
                break
            mouse = MouseController(args.mouse_precision ,args.mouse_speed)
            mouse.move(x,y)
                
        feed.close()
        cv2.destroyAllWindows()
    except RuntimeError:
        pass

def main():
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)
    

if __name__ == '__main__':
    main()