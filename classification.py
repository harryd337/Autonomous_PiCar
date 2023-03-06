import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
import pandas as pd
import random
import tensorflow_hub as hub
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"

batch_size = 1
img_height = 224
img_width = 224
seed = random.randint(1, 1000)
seed = 1

train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data',
    labels='inferred',
    label_mode='binary',
    #class_names = [0, 1],
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='training',
    crop_to_aspect_ratio = False
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data',
    labels='inferred',
    label_mode='binary',
    #class_names = [0, 1],
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='validation',
    crop_to_aspect_ratio = False
    )

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
normalization_layer = tf.keras.layers.Rescaling(1./255)

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

classifier_model = mobilenet_v2

IMAGE_SHAPE_M = (224,224)  #Image size for MobilNet
IMAGE_SHAPE_I = (299,299)  #Image size for Inception

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE_M+(3,))
])
classifier.trainable = False

model = tf.keras.Sequential([
    #normalization_layer,
    classifier, 
    layers.Dense(1)
])
model.build(input_shape=(None, ) + IMAGE_SHAPE_M)
model.summary()

#training the model

model.compile(
    optimizer = 'adam', 
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

EPOCHS = 1

history = model.fit(train_ds,
                   epochs=EPOCHS,
                   validation_data=val_ds) #change names of train_batches and validation_batches depending on oyur names