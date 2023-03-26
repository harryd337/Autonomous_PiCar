#%%
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
import random
import datetime
import tensorflow.keras as keras
from tensorflow.keras import layers, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def initialise_session():
    K.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    num_physical_devices = len(physical_devices)
    print("GPUs Available: ", num_physical_devices)
    if num_physical_devices > 0:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_files_and_paths():
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        path_to_data = Path("/content/drive/MyDrive/machine-learning-in-science-ii-2023")
        train_image_paths = pd.read_csv(str(path_to_data/'training_norm_paths_googledrive.csv'))
        test_image_paths = pd.read_csv(str(path_to_data/'test_image_paths_googledrive.csv'))
    else:
        path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
        train_image_paths = pd.read_csv(str(path_to_data/'training_norm_paths.csv'))
        test_image_paths = pd.read_csv(str(path_to_data/'test_image_paths.csv'))
    return path_to_data, train_image_paths, test_image_paths
    
    def f1_score(y_true, y_pred):
    """
    Function to calculate the F1 score.
    """
    def recall(y_true, y_pred):
        """
        Recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Precision metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score

def load_training_images_and_labels(image_path, speed_label, angle_label):
    # Define a function that maps each row to an image and a pair of labels
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_shape)
    speed_label = tf.cast(speed_label, tf.int32)
    angle_label = tf.cast(angle_label, tf.float32)
    return image, (speed_label, angle_label)

def augment(image_label, seed):
    image, label = image_label
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    return image, label
    
    def build_training_and_validation_sets():
    dataset = tf.data.Dataset.from_tensor_slices((train_image_paths['image_path'],
                                                  train_image_paths['speed'],
                                                  train_image_paths['angle']))
    dataset = dataset.map(load_training_images_and_labels).batch(batch_size)
    num_batches = dataset.cardinality().numpy()
    dataset = dataset.shuffle(buffer_size=num_batches*batch_size)

    train_batches = int(num_batches * train_val_split)
    val_batches = num_batches - train_batches

    train_set = dataset.take(train_batches)
    val_set = dataset.skip(train_batches)

    AUTOTUNE = tf.data.AUTOTUNE
    counter = tf.data.Dataset.counter()
    train_set = tf.data.Dataset.zip((train_set, (counter, counter)))
    train_set = train_set.map(augment, num_parallel_calls=AUTOTUNE)
    
    train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)
    return train_set, val_set

class CNNs(keras.Model):
    def __init__(self, image_shape, name='CNNs'):
        super(CNNs, self).__init__(name=name)
        self.image_shape = image_shape
        
        self.CNN_speed = keras.Sequential([
            Input(shape=image_shape+(3,)),
            layers.Conv2D(32, 3,
                          padding="valid",
                          activation="relu",
                          kernel_regularizer=keras.regularizers.l2(0.001)
                          ),
            layers.Dropout(0.5),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3,
                          activation="relu",
                          kernel_regularizer=keras.regularizers.l2(0.001)
                          ),
            layers.Dropout(0.5),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3,
                          activation="relu",
                          kernel_regularizer=keras.regularizers.l2(0.005)
                          ),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(64,
                         activation="relu",
                         kernel_regularizer=keras.regularizers.l2(0.01)
                         ),
            layers.Dropout(0.5),
            layers.Dense(10)
        ], name='CNN_speed')
        
        self.CNN_angle = keras.Sequential([
            Input(shape=image_shape+(3,)),
            layers.Conv2D(32, 3, padding="valid", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10)
        ], name='CNN_angle')

        self.speed_output = keras.layers.Dense(1, activation=None, name='speed')
        self.angle_output = keras.layers.Dense(1, activation='linear', name='angle')
    
    @tf.function
    def call(self, inputs):
        x = self.CNN_speed(inputs)
        y = self.CNN_angle(inputs)
        
        speed_output = self.speed_output(x)
        angle_output = self.angle_output(y)
        return [speed_output, angle_output]


def build_model():
    inputs = keras.layers.Input(shape=image_shape+(3,))
    [speed_output, angle_output] = CNNs(image_shape)(inputs)

    # Name outputs
    speed_output = layers.Lambda(lambda x: x, name='speed')(speed_output)
    angle_output = layers.Lambda(lambda x: x, name='angle')(angle_output)

    model = keras.models.Model(inputs=inputs, outputs=[speed_output, angle_output])
    model.summary()
    return model
    
    def train_model():
    model.compile(
        optimizer = keras.mixed_precision.LossScaleOptimizer(Adam()),
        loss={
            'speed': keras.losses.BinaryCrossentropy(from_logits=True),
            'angle': keras.losses.MeanSquaredError()
        },
        loss_weights={
            'speed': 1,
            'angle': 1
        }
    )

    callback = keras.callbacks.EarlyStopping(monitor='val_speed_loss',
                                                start_from_epoch=1000,
                                                restore_best_weights=True)

    if logging:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.profiler.experimental.start(log_dir)
    
    history = model.fit(train_set,
                        epochs=epochs,
                        validation_data=val_set,
                        callbacks=[callback])
    
    if logging:
        tf.profiler.experimental.stop()
        # to view log execute: "tensorboard --logdir=logs/fit/"
    return model, history
#%%

# --- HYPERPARAMETERS ---
batch_size = 40
epochs = 100
train_val_split = 0.8
image_shape = (32, 32) # (32, 32) works well
logging = False # log using tensorboard
# -----------------------

initialise_session()
path_to_data, train_image_paths, test_image_paths = load_files_and_paths()
train_set, val_set = build_training_and_validation_sets()
model = build_model()
model, history = train_model()

lowest_val_loss = str(round(min(history.history['speed_loss']), 5))
print(f"Lowest speed loss: {lowest_val_loss}")

lowest_val_loss = str(round(min(history.history['val_speed_loss']), 5))
print(f"Lowest validation speed loss: {lowest_val_loss}")
#%%
# Plot training curve
def plot_training_curve(epoch_offset):
    loss = history.history['speed_loss']
    val_loss = history.history['val_speed_loss']
    epochs_range = range(epochs)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range[epoch_offset:], loss[epoch_offset:], label='Training Loss')
    plt.plot(epochs_range[epoch_offset:], val_loss[epoch_offset:], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Speed Loss')
    plt.show()
    
plot_training_curve(epoch_offset=5)
#%%
# --- TESTING ---
# Load the testing data in the exact same way as the training data
def load_testing_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_shape)
    image = tf.expand_dims(image, axis=0)
    return image
def build_test_set():
    test_set = tf.data.Dataset.from_tensor_slices(test_image_paths['image_path'])
    test_set = test_set.map(load_testing_images)
    return test_set

def threshold_predictions(predictions_df):
    boundary = lambda x: 1 if x > 0.5 else 0
    predictions_df['speed'] = predictions_df['speed'].apply(boundary)

    angles = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
            0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
    closest_angle_round = lambda x: angles[min(range(len(angles)),
                                            key = lambda i: abs(angles[i]-x))]
    predictions_df['angle'] = predictions_df['angle'].apply(closest_angle_round)
    return predictions_df

def make_predictions():
    predictions = model.predict(test_set)

    speed_predictions = predictions[0]
    angle_predictions = predictions[1]

    predictions_df = pd.DataFrame()
    predictions_df['image_id'] = np.arange(1, 1021)
    predictions_df['angle'] = angle_predictions
    predictions_df['speed'] = speed_predictions

    predictions_df = threshold_predictions(predictions_df)
    return predictions_df
    
def create_submission():
    predictions_df.to_csv(f"submissions/submission_speedval-{lowest_val_loss}.csv", index=False)
    
test_set = build_test_set()
predictions_df = make_predictions()
create_submission()
#%%
# --- SAVE MODEL ---
def save_tf_model():
    path_to_models = Path(__file__).parent/'models'
    tf_save_path = str(path_to_models/f"tf/{lowest_val_loss}/")
    tf.saved_model.save(model, tf_save_path)
    return path_to_models, tf_save_path

def save_tflite_model():
    # Convert the model to tflite
    tflite_model = tf.lite.TFLiteConverter.from_saved_model(tf_save_path).convert()

    tflite_save_path = path_to_models/f"tflite/{lowest_val_loss}"
    os.mkdir(tflite_save_path)
    with open(tflite_save_path/'model.tflite', 'wb') as f:
        f.write(tflite_model)

path_to_models, tf_save_path = save_tf_model()
save_tflite_model()
#%%
# TRAINING LOOP
import time
lowest_val_loss = 1
while float(lowest_val_loss) > 0.005:
    initialise_session()
    path_to_data, train_image_paths, test_image_paths = load_files_and_paths()
    train_set, val_set = build_training_and_validation_sets()
    model = build_model()
    model, history = train_model()
    lowest_val_loss = str(round(min(history.history['val_loss']), 5))
    print(f"Lowest validation loss: {lowest_val_loss}")
    plot_training_curve(epoch_offset=5)
    if float(lowest_val_loss) < 0.03:
        test_set = build_test_set()
        predictions_df = make_predictions()
        create_submission()
        path_to_models, tf_save_path = save_tf_model()
        save_tflite_model()
    time.sleep(10)
#%%
