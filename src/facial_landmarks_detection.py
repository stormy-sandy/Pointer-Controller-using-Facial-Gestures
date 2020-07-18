import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import logging as log

class Facial_Landmarks_Detection_Model:
    '''
    Class for the Facial Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions = extensions

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Check if modeld path is correct")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        
        #load the model using IECore()
        self.core = IECore()
        
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        

    def predict(self, image):
        
        
        frame = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:img_processed})
        output=outputs[0]

        l_eye, r_eye,out_image = self.draw_outputs(output, image)
            
        return out_image, l_eye, r_eye
    
    def draw_outputs(self, outputs, image):
        #get image width and hight
        h = image.shape[0]
        w = image.shape[1]
        #left eye processing
        xl,yl,xr,yr, = outputs[0][0]*w,outputs[1][0]*h,outputs[2][0]*w,outputs[3][0]*h 
        
        xleft_min,yleft_min,xleft_max,yleft_max = int(xl-10),int(yl-10),int(xl+10),int(yl+10)
        
        cv2.rectangle(image, (xleft_min, yleft_min), (xleft_max, yleft_max), (0,55,255), 2) #drawing bounding box
        
        l_eye =  image[yleft_min:yleft_max, xleft_min:xleft_max]
        
        # right eye processing
        xright_min,yright_min,xright_max,yright_max = int(xr-10),int(yr-10),int(xr+10),int(yr+10)
        
        cv2.rectangle(image, (xright_min, yright_min), (xright_max, yright_max), (0,55,255), 2) #drawing bounding box
        
        r_eye = image[yright_min:yright_max, xright_min:xright_max]

        return image,l_eye, r_eye
        
    def check_model(self, core):
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
        #Get Input shape 
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame