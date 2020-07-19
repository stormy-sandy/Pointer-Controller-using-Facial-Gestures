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
from input_feeder import InputFeeder


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
    
    parser.add_argument("--print",default=False,
                        help="Print models output on frame",action="store_true")
    
    parser.add_argument("--no_move",default=False,
                        help="Not move mouse based on gaze estimation output",action="store_true")
    
    parser.add_argument("-v","--video",default=False,
                        help="Don't show video window",action="store_true")

    return parser


def infer_on_stream(args):
    fmodel = args.fdmodel
    flmodel = args.lmdmodel
    hmodel = args.hpdmodel
    gmodel = args.gemodel
    device = args.device
    video_file = args.video
    face_detect = args.FD
    eye_detect = args.ED
    gaze_v = args.GD
    head = args.HD 
    threshold = 0.6

    start_model_load_time = time.time()

    #initailizing models
    
    face_model = face(fmodel, face_detect, device, threshold)
    facial_landmarks = facial(flmodel,eye_detect, device, threshold)
    head_pose_est = head_pose(hmodel,head, device, threshold)
    gaze_est = gaze(gmodel, gaze_v, device, threshold)

    # Loading models

    face_model.load_model()
    facial_landmarks.load_model()
    head_pose_est.load_model()
    gaze_est.load_model()
    total_model_load_time = time.time() - start_model_load_time

    if video_file != "cam":
        vtype = 'video'
    else:
        vtype = 'cam'

    counter = 0
    start_inference_time = time.time()
        
    try:
        feed=InputFeeder(input_type = vtype, input_file = video_file)
        feed.load_data()
        for batch in feed.next_batch():
            if batch is None:
                log.error('The input frame is not being read, The file is corrupted')
                exit()
            counter += 1
            frame,face_crop,detections = face_model.predict(batch)
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
            mouse = MouseController('low','fast')
            mouse.move(x,y)
                
        feed.close()
        cv2.destroyAllWindows()

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        print("The total time to load all the models is :"+str(total_model_load_time)+"sec")
        print("The total inference time of the models is :"+str(total_inference_time)+"sec")
        print("The total number of frames per second is :"+str(fps)+"fps")

    

def main():
    """
    Load the network and parse the output.
    :return: None
    """



    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)
    

if __name__ == '__main__':
    main()