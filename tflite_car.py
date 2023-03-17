#%%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path

# LOAD TFLITE MODEL
# -----------------------
model_num = 0.40676
model_path = str(Path(__file__).parent / f"models/tflite/{str(model_num)}/\
model.tflite")
# -----------------------

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
image_shape = (input_shape[1], input_shape[2])
image_channels = input_shape[3]

boundary = lambda x: 1 if x > 0.5 else 0
angles = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
          0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
closest_angle_round = lambda x: angles[min(range(len(angles)),
                                           key = lambda i: abs(angles[i]-x))]
#%%
# INFERENCE
# -----------------------
image_path = str(Path(__file__).parent / f"./machine-learning-in-science-ii-\
2023/test_data/test_data/12.png")
# -----------------------

def inference(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=image_channels)
    image = tf.image.resize(image, image_shape)
    image = tf.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    speed_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    angle_prediction = interpreter.get_tensor(output_details[1]['index'])[0][0]

    speed_prediction = boundary(speed_prediction)
    angle_prediction = closest_angle_round(angle_prediction)
    return speed_prediction, angle_prediction

speed_prediction, angle_prediction = inference(image_path)
#%%