import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork


'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class FaceDetectionModel:
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

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        detections=[]
        processed_img = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name:processed_img})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        coords = coords[0] #take the first detected face
        height,width=image.shape[0],image.shape[1]
        x_min,y_min,x_max,y_max= coords
        x_min,y_min,x_max,y_max=int(x_min * width),int(y_min * height),int(x_max * width),int(y_max * height)
        
        detected_face = image[y_min:y_max, x_min:x_max]
        detections.append([x_min, y_min, x_max, y_max])

        return detected_face, detections

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):

        image_in = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_in = np.transpose(np.expand_dims(image_in, axis=0), (0,3,1,2))

        return image_in

    def preprocess_output(self, outputs, prob_threshold):

        coords =[]
        outs = outputs[self.output_name][0][0]
        for out in outs:
            conf = out[2]
            if conf>prob_threshold:
                x_min,y_min,x_max,y_max=out[3],out[4],out[5],out[6]
                coords.append([x_min,y_min,x_max,y_max])
                
        return coords
