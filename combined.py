import sys
on_colab =  'google.colab' in sys.modules
if on_colab:
    from google.colab import drive
    drive.mount('/content/drive')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
import pandas as pd
import random
import tensorflow_hub as hub
K = tf.keras.backend

K.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
num_physical_devices = len(physical_devices)
print("GPUs Available: ", num_physical_devices)
if num_physical_devices > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 20
epochs = 6
mobilenet_v2 = True
inception_v3 = False

if mobilenet_v2:
    mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    classifier_model = mobilenet_v2
    image_shape = (224, 224)
elif inception_v3:
    inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
    image_shape = (299, 299)
    classifier_model = inception_v3

if on_colab:
    path_to_data = Path("/content/drive/MyDrive/machine-learning-in-science-ii-2023")
    image_paths_csv = pd.read_csv(str(path_to_data/'training_norm_paths_googledrive.csv'))
else:
    path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
    image_paths_csv = pd.read_csv(str(path_to_data/'training_norm_paths.csv'))

# Create a dataset from the dataframe
dataset = tf.data.Dataset.from_tensor_slices((image_paths_csv['image_path'],
                                              image_paths_csv['speed'],
                                              image_paths_csv['angle']))

# Define a function that maps each row to an image and a pair of labels
def load_image_and_labels(image_path, speed_label, angle_label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_shape)
    # Convert the class label to an integer
    speed_label = tf.cast(speed_label, tf.int32)
    # Convert the regression label to a float
    angle_label = tf.cast(angle_label, tf.float32)
    #angle_label = tf.cast(angle_label * 16, tf.int32) # multiply by 16 to get values between 0 and 16
    #angle_label = tf.one_hot(angle_label, depth=17)
    return image, (speed_label, angle_label)

dataset = dataset.map(load_image_and_labels).batch(batch_size) # Apply the function to each batch of data
num_batches = dataset.cardinality().numpy() # Calculate the number of batches in the dataset

# Shuffle the data using a buffer size equal to or larger than the number of elements in the dataset
dataset = dataset.shuffle(buffer_size=num_batches*batch_size)

# Calculate the number of batches for training and validation
train_batches = int(num_batches * 0.8)
val_batches = num_batches - train_batches

# Split the dataset into training and validation sets using take and skip methods
train_set = dataset.take(train_batches)
val_set = dataset.skip(train_batches)

AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)

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

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=image_shape+(3,))
])
classifier.trainable = False

inp = tf.keras.layers.Input((image_shape[0], image_shape[1], 3)) #RGB images of size (x, y)

#add the classifier layers before the dense layers
x = classifier(inp) #assuming classifier is a layer or a model

#you can define multiple outputs from x
classification_output = tf.keras.layers.Dense(1, activation=None, name='speed')(x) #binary classification
regression_output = tf.keras.layers.Dense(1, activation='linear', name='angle')(x) #regression

#create a model with one input and two outputs
model = tf.keras.models.Model(inputs=inp, outputs=[classification_output, 
                                                   regression_output
                                                   ])

model.summary()

#train the model

model.compile(
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam()),
    loss={
        'speed': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        'angle': tf.keras.losses.MeanSquaredError()
        #'angle': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    },
    metrics={
        'speed': ['accuracy', f1_score],
        'angle': 'mse'
    },
    loss_weights={
        'speed': 0.5,
        'angle': 0.5
    }
)

history = model.fit(train_set,
                   epochs=epochs,
                   validation_data=val_set)