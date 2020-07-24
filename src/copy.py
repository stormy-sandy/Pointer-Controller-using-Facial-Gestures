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
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0] + '.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        ie = IECore()
        net = ie.read_network(model=self.model_structure, weights=self.model_weights)
        # exec_net = ie.load_network(network=net, device_name="CPU", num_requests=1)

        self.plugin = ie
        self.network = net
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0 and self.device == 'CPU':
            print(f"Unsupported layers found:{unsupported_layers}")
            if not self.extensions == None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers) != 0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Provide the path to cpu extension")
                exit(1)

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_img = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name: processed_img})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords) == 0):
            return 0, 0
        coords = coords[0]  # take the first detected face
        h = image.shape[0]
        w = image.shape[1]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        detected_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return detected_face, coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):

        image_res = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_res = np.transpose(np.expand_dims(image_res, axis=0), (0, 3, 1, 2))

        return image_res

    def preprocess_output(self, outputs, prob_threshold):

        coords = []
        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            if conf > prob_threshold:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coords.append([x_min, y_min, x_max, y_max])

        return coords
