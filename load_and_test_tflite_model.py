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

# TEST ON TESTING DATASET
import pandas as pd
import numpy as np

path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
test_image_paths_csv = pd.read_csv(str(path_to_data/'test_image_paths.csv'))
test_set = tf.data.Dataset.from_tensor_slices(test_image_paths_csv['image_path'])

def load_testing_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_shape)
    image = tf.expand_dims(image, axis=0)
    return image

test_set = test_set.map(load_testing_images)

images = []
for image in test_set:
    images.append(image)

num_images = len(images)
speed_predictions = np.zeros(num_images)
angle_predictions = np.zeros(num_images)
for i, image in enumerate(images):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    speed_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    angle_prediction = interpreter.get_tensor(output_details[1]['index'])[0][0]

    speed_predictions[i] = boundary(speed_prediction)
    angle_predictions[i] = closest_angle_round(angle_prediction)
    #angle_predictions[i] = angle_prediction

predictions_df = pd.DataFrame()
predictions_df['image_id'] = np.arange(1, 1021)
predictions_df['angle'] = angle_predictions
predictions_df['speed'] = speed_predictions

predictions_df.to_csv(f"loaded_tflite_model_results.csv", index=False)