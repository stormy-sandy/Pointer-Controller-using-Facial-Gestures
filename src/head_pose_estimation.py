import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork


'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class HeadPoseEstimationModel:
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
        final_output = self.preprocess_output(outputs)

        return final_output


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):

        image_res = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_res = np.transpose(np.expand_dims(image_res, axis=0), (0,3,1,2))

        return image_res

    def preprocess_output(self, outputs):

        outs = []
        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])

        return outs
