import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork
import math
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class GazeEstimationModel:
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

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image, right_eye_image)
        outputs = self.net.infer({'head_pose_angles':head_pose_angles, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs, head_pose_angles)

        return new_mouse_coord, gaze_vector
        


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left_eye_image, right_eye_image):

        le_image_res,re_image_res = cv2.resize(left_eye_image, (60, 60)),cv2.resize(right_eye_image, (60,60))
        le_img_processed,re_img_processed = np.transpose(np.expand_dims(le_image_res, axis=0), (0,3,1,2)) , np.transpose(np.expand_dims(re_image_res, axis=0), (0,3,1,2))

        return le_img_processed, re_img_processed


    def preprocess_output(self, outputs, head_pose_angles):

        gaze_vector = outputs[self.output_name][0]
        mouse_cord = (0, 0)
        try:
            angle_r_fc = head_pose_angles[2]
            sin_r,cos_r = math.sin(angle_r_fc * math.pi / 180.0), math.cos(angle_r_fc * math.pi / 180.0)
            x,y = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r, -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
             
            mouse_cord = (x, y)
        except Exception as e:
            self.logger.error("Error While preprocessing output in Gaze Estimation Model" + str(e))
        return mouse_cord, gaze_vector