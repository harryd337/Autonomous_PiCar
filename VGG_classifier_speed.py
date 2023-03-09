import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
import pandas as pd
import random
import tensorflow_hub as hub
from keras import layers
import keras.backend as K
from keras.optimizers import Adam

physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", len(physical_devices))
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

!nvidia-smi

from google.colab import drive
drive.mount("/Shareddrives/")

path_to_data = Path("/Shareddrives/MyDrive/machine-learning-in-science-ii-2023")

batch_size = 20
img_height = 224
img_width = 224
seed = random.randint(1, 1000)

train_set = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='training',
    crop_to_aspect_ratio = False
    )
    
val_set = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data',
    labels='inferred',
    label_mode='binary',
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
train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)
    
#normalization_layer = tf.keras.layers.Rescaling(1./255)

#mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
#inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

#classifier_model = mobilenet_v2

#IMAGE_SHAPE_M = (224,224)  #Image size for MobilNet
#IMAGE_SHAPE_I = (299,299)  #Image size for Inception
#IMAGE_SHAPE_V = (224, 224) #Image size for VGG

#classifier = tf.keras.Sequential([
    #hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE_M+(3,))
#])
#classifier.trainable = False

#model = tf.keras.Sequential([
    #classifier, 
    #layers.Dense(1,
                 #activation='relu'
                 #kernel_regularizer=tf.keras.regularizers.l2(0.1)
                 #)
#])
#model.summary()

from pickle import FALSE
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGE_SHAPE_V = (224,224)
standard_model = VGG16(weights="imagenet",include_top=False, input_shape=IMAGE_SHAPE_V+(3,))
standard_model.trainable=FALSE

#train_set = preprocess_input(train_set)
#val_set = preprocess_input(val_set)

standard_model.summary()

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(1, activation='softmax')

model = tf.keras.Sequential([
    standard_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
]) 

from tensorflow.keras.callbacks import EarlyStopping

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

#if using early stopping
#E_S = EarlyStopping(monitor='val_accuracy', model'max',patience=5, restore_best_weights=True)
#model.fit(training_data, train_labels, epochs=100, validation=0.2, batch_size=32, callbacks=[es])

#otherwise us this
EPOCHS = 3
history = model.fit(train_set,
                   epochs=EPOCHS,
                   validation_data=val_set)  #replace validation batches
