import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore

import os
import cv2
import argparse
import sys
class face_det_Model:
    '''
    Class for the face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model = self.core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
    def predict(self, image):
        p_frame = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name: p_frame})
        coords = self.preprocess_outputs(outputs[self.output_name])
        self.draw_outputs(coords, image)
        
        post_image,post_coord=self.draw_outputs(coords, image)
        return post_image,post_coord
    
    def draw_outputs(self, coords, image):
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
                
                
        

    def preprocess_outputs(self, outputs):
        cords = []
        for box in outputs[0][0]: # output.shape: 1x1xNx7
            thresh = box[2]
            if thresh >= self.threshold:
                xmin,ymin,xmax,ymax = box[3],box[4],box[5],box[6] 
                cords.append((xmin, ymin, xmax, ymax))
                
                
        return cords

    def preprocess_input(self, image):
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame