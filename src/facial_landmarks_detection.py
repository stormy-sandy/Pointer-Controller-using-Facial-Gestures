import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class FacialLandmarksDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions

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
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_img = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name:processed_img})
        coords = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) #(lefteye_x, lefteye_y, righteye_x, righteye_y)
        xl_axis=coords[0]
        yl_axis=coords[1]
        xr_axis=coords[2]
        yr_axis=coords[3]
        lefti_xmin=xl_axis-10
        lefti_ymin=yl_axis-10
        lefti_xmax=xl_axis+10
        lefti_ymax=yl_axis+10
        
        righti_xmin=xr_axis-10
        righti_ymin=yr_axis-10
        righti_xmax=xr_axis+10
        righti_ymax=yr_axis+10

        left_eye,right_eye =  image[lefti_ymin:lefti_ymax, lefti_xmin:lefti_xmax],image[righti_ymin:righti_ymax, righti_xmin:righti_xmax]
        eye_coords = [[lefti_xmin,lefti_ymin,lefti_xmax,lefti_ymax], [righti_xmin,righti_ymin,righti_xmax,righti_ymax]]

        return left_eye, right_eye, eye_coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_res = cv2.resize(image_rgb, (self.input_shape[3], self.input_shape[2]))
        image_res = np.transpose(np.expand_dims(image_res, axis=0), (0,3,1,2))
        
        return image_res

    def preprocess_output(self, outputs):

        outs = outputs[self.output_name][0]
        leye_x,leye_y,reye_x,reye_y  = outs[0].tolist()[0][0],outs[1].tolist()[0][0],outs[2].tolist()[0][0],outs[3].tolist()[0][0]
        return (leye_x, leye_y, reye_x, reye_y)
