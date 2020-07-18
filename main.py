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
    
    parser.add_argument("--no_video",default=False,
                        help="Don't show video window",action="store_true")

    return parser


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("gaze-app.log"),
                logging.StreamHandler()
            ])
        

        # Initialise the class
        mc = MouseController("low","fast")
        #mc.move(100,100)
        fd = face_det_Model(args.fdmodel)
        lmd = Facial_Landmarks_Detection_Model(args.lmdmodel)
        hpd = Head_pose_Model(args.hpdmodel)
        ge = Gaze_est_Model(args.gemodel)

        ### Load the model through ###
        logging.info("============== Models Load time ===============") 
        start_time = time.time()
        fd.load_model()
        logging.info("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        start_time = time.time()
        lmd.load_model()
        logging.info("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        start_time = time.time()
        hpd.load_model()
        logging.info("Headpose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        start_time = time.time()
        ge.load_model()
        logging.info("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )
        logging.info("==============  End =====================") 
        # Get and open video capture
        feeder = InputFeeder('video', args.input)
        feeder.load_data()
        # FPS = feeder.get_fps()

        # Grab the shape of the input 
        # width = feeder.get_width()
        # height = feeder.get_height()

        # init scene variables
        frame_count = 0

        ### Loop until stream is over ###
        fd_infertime = 0
        lm_infertime = 0
        hp_infertime = 0
        ge_infertime = 0
        while True:
            # Read the next frame
            try:
                frame = next(feeder.next_batch())
            except StopIteration:
                break

            key_pressed = cv2.waitKey(60)
            frame_count += 1
                      
            # face detection
            
            face_frame,bboxes = fd.predict(frame) #cropped face with bounding box co-ords
            
            
            
            #for each face
            for fbox in bboxes:
                
                
                # get face landmarks
                # crop face from frame
                face = frame[fbox[1]:fbox[3],fbox[0]:fbox[2]]
                p_frame = lmd.preprocess_input(face)
                
                start_time = time.time()
                lmoutput = lmd.predict(p_frame)
                lm_infertime += time.time() - start_time
                out_frame,left_eye_point,right_eye_point = lmnet.preprocess_output(lmoutput, fbox, out_frame,args.print)

                # get head pose estimation
                p_frame  = hpnet.preprocess_input(face)
                start_time = time.time()
                hpoutput = hpnet.predict(p_frame)
                hp_infertime += time.time() - start_time
                out_frame, headpose_angels = hpnet.preprocess_output(hpoutput,out_frame, face,fbox,args.print)

                # get gaze  estimation
                out_frame, left_eye, right_eye  = genet.preprocess_input(out_frame,face,left_eye_point,right_eye_point,args.print)
                start_time = time.time()
                geoutput = genet.predict(left_eye, right_eye, headpose_angels)
                ge_infertime += time.time() - start_time
                out_frame, gazevector = genet.preprocess_output(geoutput,out_frame,fbox, left_eye_point,right_eye_point,args.print)

                if(not args.no_video):
                    cv2.imshow('im', out_frame)
                
                if(not args.no_move):
                    mc.move(gazevector[0],gazevector[1])
                
                #consider only first detected face in the frame
                break
            
            # Break if escape key pressed
            if key_pressed == 27:
                break

        #logging inference times
        if(frame_count>0):
            logging.info("============== Models Inference time ===============") 
            logging.info("Face Detection:{:.1f}ms".format(1000* fd_infertime/frame_count))
            logging.info("Facial Landmarks Detection:{:.1f}ms".format(1000* lm_infertime/frame_count))
            logging.info("Headpose Estimation:{:.1f}ms".format(1000* hp_infertime/frame_count))
            logging.info("Gaze Estimation:{:.1f}ms".format(1000* ge_infertime/frame_count))
            logging.info("============== End ===============================") 

        # Release the capture and destroy any OpenCV windows
        feeder.close()
        cv2.destroyAllWindows()
    except Exception as ex:
        logging.exception("Error in inference:" + str(ex))
        

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