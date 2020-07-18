'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore

import os
import cv2
import argparse
import sys
class face_det_Model:
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

    def draw_outputs(self,coords, image):
        """
        Draws outputs or predictions on image.

        """

        width = image.shape[1]
        height = image.shape[0]
        box=[]
        for ob in coords:
                # Draw bounding box for object when it's probability is more than
                #  the specified threshold
                
                box_side_1=(int(ob[0] * width),int(ob[1] * height))
                    
                box_side_2=(int(ob[2] * width),int(ob[2] * height))
                    # Write out the frame
                    
                    
                cv2.rectangle(image, box_side_1,box_side_2, (0, 55, 255), 1)
                box.append([box_side_1[0], box_side_1[1], box_side_2[0], box_side_2[1]])
                
        return box, image

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(image)
        # Start asynchronous inference for specified request
        result= self.net.infer(inputs={self.input_name:p_frame})
        outputs = result[self.output_name]
        coords = self.preprocess_output(outputs)
        b_box, image = self.draw_outputs(coords, image)
        cropd_face = image[b_box[1]:b_box[3], b_box[0]:b_box[2]]
        return cropd_face, b_box



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
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame


    def preprocess_output(self, outputs):
        '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        cords = []
        for box in outputs[0][0]: # output.shape: 1x1xNx7
            thresh = box[2]
            if thresh >= self.threshold:
                xmin,ymin,xmax,ymax = box[3],box[4],box[5],box[6] 
                cords.append((xmin, ymin, xmax, ymax))
                
                
        return cords

