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
from tensorflow.keras import layers, Input
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

# Define a function that maps each row to an image and a pair of labels
def load_training_images_and_labels(image_path, speed_label, angle_label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_shape)
    speed_label = tf.cast(speed_label, tf.int32)
    angle_label = tf.cast(angle_label, tf.float32)
    return image, (speed_label, angle_label)

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
    train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)
    return train_set, val_set


class CNNs(tf.keras.Model):
    def __init__(self, image_shape, name='CNNs'):
        super(CNNs, self).__init__(name=name)
        self.image_shape = image_shape
        
        self.shared_augment = tf.keras.Sequential([
            layers.Lambda(lambda x:
                self.random_brightness(x, max_delta=0.2), input_shape=image_shape+(3,))
        ])
        
        # speed_augment = tf.keras.Sequential([
        #     layers.RandomFlip("horizontal_and_vertical"),
        #     layers.RandomRotation(0.2)
        #     ])
        
        # angle_augment = tf.keras.Sequential([
        #     layers.RandomFlip("horizontal_and_vertical"),
        #     layers.RandomRotation(0.2)
        #     ])
        
        self.CNN_speed = tf.keras.Sequential([
            Input(shape=image_shape+(3,)),
            #speed_augment,
            layers.Conv2D(32, 3, padding="valid", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ], name='CNN_speed')
        
        self.CNN_angle = tf.keras.Sequential([
            Input(shape=image_shape+(3,)),
            #angle_augment,
            layers.Conv2D(32, 3, padding="valid", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ], name='CNN_angle')

        self.speed_output = tf.keras.layers.Dense(1, activation=None, name='speed')
        self.angle_output = tf.keras.layers.Dense(1, activation='linear', name='angle')
    
    def random_brightness(self, x, max_delta):
        seed = tf.random.experimental.stateless_split(
            tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32),
            num=1)[0]
        return tf.map_fn(lambda image:
            tf.image.stateless_random_brightness(image, max_delta=max_delta, seed=seed), x)
    
    @tf.function
    def call(self, inputs, training=False):
        if training:
            z = self.shared_augment(inputs)
        else:
            z = inputs
        x = self.CNN_speed(z)
        y = self.CNN_angle(z)
        
        speed_output = self.speed_output(x)
        angle_output = self.angle_output(y)
        return [speed_output, angle_output]


def build_model():
    inputs = tf.keras.layers.Input(shape=image_shape+(3,))
    [speed_output, angle_output] = CNNs(image_shape)(inputs)

    # Name outputs
    speed_output = layers.Lambda(lambda x: x, name='speed')(speed_output)
    angle_output = layers.Lambda(lambda x: x, name='angle')(angle_output)

    model = tf.keras.models.Model(inputs=inputs, outputs=[speed_output, angle_output])
    model.summary()
    return model

def train_model():
    model.compile(
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam()),
        loss={
            'speed': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'angle': tf.keras.losses.MeanSquaredError()
        },
        metrics={
            'speed': f1_score
        },
        loss_weights={
            'speed': 1,
            'angle': 1
        }
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=min_delta,
                                                patience=patience,
                                                baseline=baseline,
                                                start_from_epoch=start_from_epoch,
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
epochs = 50
train_val_split = 0.8
image_shape = (32, 32)
logging = False # log using tensorboard
# Early stopping:
min_delta = 0.005
patience = 0
baseline = None
start_from_epoch = 50
# -----------------------

initialise_session()
path_to_data, train_image_paths, test_image_paths = load_files_and_paths()
train_set, val_set = build_training_and_validation_sets()
model = build_model()
model, history = train_model()

lowest_val_loss = str(round(min(history.history['val_loss']), 5))
print(f"Lowest validation loss: {lowest_val_loss}")
#%%
# Plot training curve
def plot_training_curve(epoch_offset):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range[epoch_offset:], loss[epoch_offset:], label='Training Loss')
    plt.plot(epochs_range[epoch_offset:], val_loss[epoch_offset:], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
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

def threshold_predictions():
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

    predictions_df = threshold_predictions()
    return predictions_df
    
def create_submission():
    predictions_df.to_csv(f"submissions/submission-{lowest_val_loss}.csv", index=False)
    
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