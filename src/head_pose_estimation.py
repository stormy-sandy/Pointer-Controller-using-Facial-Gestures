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
class Model_X:
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

    def draw_outputs(self, image):
        """
        Draws outputs or predictions on image.

        Args:
            coords: The coordinates of predictions.
            image: The image on which boxes need to be drawn.

        Returns:
            the frame
            the count of people
            bounding boxes above threshold
        """

        initial_h = image.shape[0]
        initial_w = image.shape[1]
        bounding_box = []
        for value in coords:
            # Draw bounding box on detected objects
            xmin = int(value[3] * initial_w)
            ymin = int(value[4] * initial_h)
            xmax = int(value[5] * initial_w)
            ymax = int(value[6] * initial_h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 2)
            bounding_box.append([xmin, ymin, xmax, ymax])
        return bounding_box, image

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)
        # Start asynchronous inference for specified request
        cropped_image = image
        net.start_async(request_id=0, inputs={self.input_name: processed_image})
        # Wait for the result
        if net.requests[0].wait(-1) == 0:
            # get out put
            outputs = net.requests[0].outputs[self.output_name]
            coords = self.preprocess_output(outputs)
            bounding_box, image = self.draw_outputs(coords, image)
            bounding_box = bounding_box[0]
            cropped_image = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        return cropped_image



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
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        """
        input_img = image

        n, c, h, w = self.input_shape

        input_img = cv2.resize(input_img, (w, h), interpolation=cv2.INTER_AREA)

        # Change image from HWC to CHW
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape((n, c, h, w))

        return input_img


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

