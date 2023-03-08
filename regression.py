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

# If running locally:
#path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"

# If running on colab:
path_to_data = Path("/content/drive/My Drive/machine-learning-in-science-ii-2023")

batch_size = 20
image_shape = (224, 224)
seed = random.randint(1, 1000)

train_set = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data/regression',
    labels='inferred',
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
    path_to_data/'training_data/regression',
    labels='inferred',
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

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

classifier_model = mobilenet_v2

IMAGE_SHAPE_M = (224,224)  #Image size for MobilNet
IMAGE_SHAPE_I = (299,299)  #Image size for Inception

IMAGE_SHAPE = IMAGE_SHAPE_M

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])
classifier.trainable = False

model = tf.keras.Sequential([
    classifier, 
    layers.Dense(1,
                 activation='linear'
                 #kernel_regularizer=tf.keras.regularizers.l2(0.1)
                 )
])

model.summary()

#training the model

model.compile(
    optimizer = Adam(),
    loss = tf.keras.losses.MeanSquaredError(),
    )

EPOCHS = 2

history = model.fit(train_set,
                   epochs=EPOCHS,
                   validation_data=val_set)