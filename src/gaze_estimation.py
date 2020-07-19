'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import logging as log
import os
import cv2
import argparse
import sys
class Gaze_est_Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        '''
        TODO: Use this to set your instance variables.
        '''


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        
    def predict(self, left_eye,right_eye,h_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        r_frame=self.preprocess_input(right_eye)
        l_frame=self.preprocess_input(left_eye)
        results= self.net.infer(inputs={"head_pose_angles": h_pose_angles,
                                        "left_eye_image":   l_frame,
                                        "right_eye_image":  r_frame})


        x,y = results['gaze_vector'][0][0],results['gaze_vector'][0][1]

        if self.gaze == "gaze_vector":
            print(results)
        return x,y



    def check_model(self):

        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            core.add_extension(self.extensions, self.device)

        ###: Check for supported layers ###
        if "CPU" in self.device:
            supported_layers = core.query_network(self.model, "CPU")
            not_supported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(self.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
                sys.exit(1)


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        '''
        input_img = image

        p_frame = cv2.resize(image, (60,60)
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame



    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        bounding_box = []

        for value in outputs[0][0]:
            # check if confidence is greater than probability threshold
            if value[2] > self.threshold:
                bounding_box.append(value)
        return bounding_box

