import tensorflow.lite as tflite
from tensorflow.image import resize
from tensorflow import expand_dims
import os
import time
import traceback

class Model:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #enable gpu
    my_model = 'kaggle-0.03653.tflite'

    def __init__(self):
        try: #load edge TPU model
            delegate = tflite.experimental.load_delegate('libedgetpu.so.1') #'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
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
        image = resize(image, self.image_shape)
        image = expand_dims(image, axis=0)
        return image

    def predict(self, image):
        image = self.preprocess(image)

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        
        speed_prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        angle_prediction = self.interpreter.get_tensor(self.output_details[1]['index'])[0][0]
        
        
        speed_prediction = self.apply_speed_threshold(speed_prediction)
        
        speed_prediction = speed_prediction*35
        angle_prediction = (angle_prediction*80 + 50).astype(int)
        
        return angle_prediction, speed_prediction
