import cv2
import os
from pysimplelog import Logger
import numpy as np

from src.face_detection import FaceDetectionModel
from src.facial_landmarks_detection import FacialLandmarksDetectionModel
from src.gaze_estimation import GazeEstimationModel
from src.head_pose_estimation import HeadPoseEstimationModel
from src.mouse_controller import MouseController
from src.input_feeder import InputFeeder

from argparse import ArgumentParser

import time

def build_argparser():
    #Parse command line arguments.

    #:return: command line arguments
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--face_detection_model", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fldm", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmarks Detection model.")
    parser.add_argument("-hpem", "--head_pose_estimation_model", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-gem", "--gaze_estimation_model", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--visualization_flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Separate each flag by space)"
                             "for seeing the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("--cpu_extension", required=False, type=str,
                        default=None,
                        help="CPU Extension for custom layers")
    parser.add_argument("--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold to be used by the model.")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Specify the target device to perform inference on: "
                             "CPU, GPU, FPGA, or MYRIAD")
    parser.add_argument("-m_prec", "--mouse_precision", type=str, default='high',
                        help="(Optional) Specify mouse precision (how much the mouse moves): 'high', 'medium', 'low'."
                             "Default is high.")
    parser.add_argument("-m_speed", "--mouse_speed", type=str, default='immediate',
                        help="(Optional) Specify mouse speed (how many secs before it moves): 'immediate'(0s), 'fast'(0.1s),"
                             "'medium'(0.5s) and 'slow'(1s). Default is immediate.")
    
    return parser



def main():

    # Grab command line arguments
    args = build_argparser().parse_args()
    flags = args.visualization_flags
    
    l =Logger("Model_stats")
    l.set_log_file_basename("Project_Logs")
    input_file_path = args.input
    feeder = None
    if input_file_path.lower() == "cam":
            feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path):
            l.error("Unable to find input video file")
            exit(1)
        feeder = InputFeeder("video", input_file_path)
    
    
    
    fdm = FaceDetectionModel(args.face_detection_model, args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(args.facial_landmarks_detection_model, args.device, args.cpu_extension)
    gem = GazeEstimationModel(args.gaze_estimation_model, args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(args.head_pose_estimation_model, args.device, args.cpu_extension)
    mc = MouseController(precision=args.mouse_precision,speed=args.mouse_speed)
    
    feeder.load_data()
    model_load_time_start=time.time()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    total_model_load_time=time.time()-model_load_time_start
    l.info("Model load time : "+str(total_model_load_time))
    
    inference_time=time.time()
    frame_count = 0
    for ret, frame in feeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video', cv2.resize(frame, (500,500)))
    
        key = cv2.waitKey(60)
        crop_face, face_coords = fdm.predict(frame, args.prob_threshold)
        if type(crop_face)==int:
            l.info("Unable to detect the face.")
            if key==27:
                break
            continue
        
        hp_output = hpem.predict(crop_face)
        
        left_eye, right_eye, eye_coords = fldm.predict(crop_face)
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_output)
        
        if (not len(flags)==0):
            preview_frame = frame
            l_eye=eye_coords[0]
            r_eye=eye_coords[1]

            if 'fd' in flags:
                preview_frame = crop_face
            if 'fld' in flags:
                l_eye=eye_coords[0]
                r_eye=eye_coords[1]
                cv2.rectangle(crop_face, (l_eye[0]-10, l_eye[1]-10), (l_eye[2]+10, l_eye[3]+10), (0,255,0), 3)
                cv2.rectangle(crop_face, (r_eye[0]-10, r_eye[1]-10), (r_eye[2]+10, r_eye[3]+10), (0,255,0), 3)
                
            if 'hp' in flags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_output[0], hp_output[1], hp_output[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in flags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le = cv2.line(left_eye, (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye, (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                crop_face[l_eye[1]:l_eye[3],l_eye[0]:l_eye[2]] = le
                crop_face[r_eye[1]:r_eye[3],r_eye[0]:r_eye[2]] = re
                
            cv2.imshow("Preview", cv2.resize(preview_frame,(500,500)))
        
        if frame_count%5==0:
            mc.move(new_mouse_coord[0], new_mouse_coord[1])    
        if key==27:
                break
    l.info("Total Inference time : "+str(time.time()-inference_time)+" Seconds")
    l.info("Stream Over")
    cv2.destroyAllWindows()
    feeder.close()
     
    

if __name__ == '__main__':
    main()