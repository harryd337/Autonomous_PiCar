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
import datetime
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
epochs = 2
mobilenet_v2 = True
inception_v3 = False
seed = random.randint(1, 1000)

if mobilenet_v2:
    mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    classifier_model = mobilenet_v2
    image_shape = (224, 224)
elif inception_v3:
    inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
    image_shape = (299,299)
    classifier_model = inception_v3

if on_colab:
    path_to_data = Path("/content/drive/My Drive/machine-learning-in-science-ii-2023")
else:
    path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"

train_set = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data/training_data/classification',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_shape,
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='training',
    crop_to_aspect_ratio = False
    )

val_set = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data/training_data/classification',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_shape,
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='validation',
    crop_to_aspect_ratio = False
    )

AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=image_shape+(3,))
])
classifier.trainable = False

model = tf.keras.Sequential([
    classifier, 
    tf.keras.layers.Dense(1,
                 activation='relu' #try with relu/None
                 #kernel_regularizer=tf.keras.regularizers.l2(0.1)
                 )
])

model.summary()

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

#training the model

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy', f1_score]
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch=(30,50))

history = model.fit(train_set,
                   epochs=epochs,
                   validation_data=val_set,
                   callbacks=[tensorboard_callback])