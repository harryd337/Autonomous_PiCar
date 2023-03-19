#%%
# --- HYPERPARAMETERS ---
model_num = 0.02477
image_shape = (32, 32)
# -----------------------
#%%
# --- TF ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow.lite as tflite
from tensorflow.io import read_file
from tensorflow.image import decode_png, resize
from tensorflow import expand_dims

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

save_path = Path(__file__).parent /f"models/tf/{str(model_num)}/"
loaded_model = tf.saved_model.load(save_path)

images = []
for image in test_set:
    images.append(image)

num_images = len(images)
speed_predictions = np.zeros(num_images)
angle_predictions = np.zeros(num_images)
for i, image in enumerate(images):
    prediction = loaded_model(image)
    speed_predictions[i] = tf.get_static_value(prediction[0])[0][0]
    angle_predictions[i] = tf.get_static_value(prediction[1])[0][0]

boundary = lambda x: 1 if x > 0.5 else 0
speed_predictions = np.vectorize(boundary)(speed_predictions)

angles = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
          0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
closest_angle_round = lambda x: angles[min(range(len(angles)),
                                           key = lambda i: abs(angles[i]-x))]
angle_predictions = np.vectorize(closest_angle_round)(angle_predictions)

predictions_df = pd.DataFrame()
predictions_df['image_id'] = np.arange(1, 1021)
predictions_df['angle'] = angle_predictions
predictions_df['speed'] = speed_predictions

predictions_df.to_csv(f"loaded_model_results.csv", index=False)
#%%
# --- TFLITE ---

# LOAD TFLITE MODEL
model_path = str(Path(__file__).parent / f"models/tflite/{str(model_num)}/\
model.tflite")

interpreter = tflite.Interpreter(model_path=model_path)
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

def inference(image_path):
    image = read_file(image_path)
    image = decode_png(image, channels=image_channels)
    image = resize(image, image_shape)
    image = expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    speed_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    angle_prediction = interpreter.get_tensor(output_details[1]['index'])[0][0]

    speed_prediction = boundary(speed_prediction)
    angle_prediction = closest_angle_round(angle_prediction)
    return speed_prediction, angle_prediction

# INFERENCE
speed_predictions = []
angle_predictions = []

for i in range(1020):
    image_path = str(Path(__file__).parent / f"./machine-learning-in-science-ii-\
2023/test_data/test_data/{i+1}.png")
    speed_prediction, angle_prediction = inference(image_path)
    speed_predictions.append(speed_prediction)
    angle_predictions.append(angle_prediction)

predictions_df = pd.DataFrame()
predictions_df['image_id'] = np.arange(1, 1021)
predictions_df['angle'] = angle_predictions
predictions_df['speed'] = speed_predictions

predictions_df.to_csv(f"loaded_tflite_model_results.csv", index=False)
#%%
# --- COMPARE MODELS ---
def calc_mae(column1, column2):
    mae = np.mean(np.abs(column1 - column2))
    return mae

def calc_average_percentage_error(mae, column):
    # calculate the average percentage error per data point
    average_percentage_error = (mae/np.mean(column))*100
    return average_percentage_error

def compare_predictions(predictions_csv1, predictions_csv2):
    loaded_csv1 = pd.read_csv(predictions_csv1)
    loaded_csv2 = pd.read_csv(predictions_csv2)

    angle1 = loaded_csv1['angle']
    angle2 = loaded_csv2['angle']

    speed1 = loaded_csv1['speed']
    speed2 = loaded_csv2['speed']

    mae_angle = calc_mae(angle1, angle2)
    mae_speed = calc_mae(speed1, speed2)

    average_percentage_error_angle = calc_average_percentage_error(mae_angle, angle1)
    average_percentage_error_speed = calc_average_percentage_error(mae_speed, speed1)
    print('Average percentage error')
    print(f"Angle: {round(average_percentage_error_angle, 4)}%")
    print(f"Speed: {round(average_percentage_error_speed, 4)}%")
    
compare_predictions('loaded_model_results.csv', 'loaded_tflite_model_results.csv')
#%%