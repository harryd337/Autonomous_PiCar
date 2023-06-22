# This script is to be loaded onto the RPi on the car to interface with its
# pre-installed software.

import os
import time
import traceback

import tensorflow.lite as tflite
from tensorflow import expand_dims
from tensorflow.image import resize


class Model:
    """Loaded TensorFlow Lite model able to interface with the cars software.

    Attributes:
        my_model (str): name of model file.
        interpreter (tflite.Interpreter): interpreter interface for running
        TensorFlow Lite models.
        input_details (list): a list in which each item is a dictionary with
        details about an input tensor.
        output_details (list): a list in which each item is a dictionary with
        details about an output tensor.
        image_shape (tuple): shape the image is to be resized to.
        apply_speed_threshold (function): replaces input with 1 if input is
        greater than "speed_threshold". Replaces input with 0 if not.
    """
    
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    my_model = 'v2_all-data_0.01923.tflite'

    def __init__(self):
        """Loads model and attempts to delegate interpreter processes to TPU.
        
        If delegation to TPU fails, falls back to CPU.
        """
        
        try:
            delegate = tflite.experimental.load_delegate('libedgetpu.so.1')
            # 'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
            print('Using TPU')
            self.interpreter = tflite.Interpreter(model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.my_model), experimental_delegates=[delegate])                                                           
        except (ValueError, OSError) as e:
            print('Fallback to CPU')
            print(traceback.format_exc())
            self.interpreter = tflite.Interpreter(model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.my_model))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = self.input_details[0]['shape']
        self.image_shape = (input_shape[1], input_shape[2])
        self.apply_speed_threshold = lambda x: 1 if x > 0.5 else 0
        
    def preprocess(self, image):
        """Resizes the image and prepares it to be input to the model.

        Args:
            image (numpy.ndarray): image captured by the camera.

        Returns:
            image (EagerTensor): resized image ready for the model.
        """
        
        image = resize(image, self.image_shape)
        image = expand_dims(image, axis=0)
        return image

    def predict(self, image):
        """Uses model to make predictions on image. Applies thresholding.
        
        Preprocesses image. Feeds image to model which makes predictions. 
        Applies speed threshold to the speed prediction. Unnormalises
        predictions.

        Args:
            image (numpy.ndarray): image captured by the camera.

        Returns:
            angle_prediction (int): unnormalised prediction of angle.
            speed_prediction (int): unnormalised prediction of speed.
        """
        
        image = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        speed_prediction = self.interpreter.get_tensor(
            self.output_details[0]['index'])[0][0]
        angle_prediction = self.interpreter.get_tensor(
            self.output_details[1]['index'])[0][0]
        speed_prediction = self.apply_speed_threshold(speed_prediction)
        speed_prediction = speed_prediction*35
        angle_prediction = (angle_prediction*80 + 50).astype(int)
        return angle_prediction, speed_prediction