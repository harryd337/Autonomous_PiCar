#%%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
import pandas as pd
import random
import matplotlib.pyplot as plt

path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
training_norm = pd.read_csv(path_to_data/'training_norm.csv')

#%%
batch_size = 32
img_height = 320
img_width = 240
seed = random.randint(1, 1000)

train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data',
    labels=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='training',    
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_data/'training_data',
    labels=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='validation',    
    )

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

plt.figure(figsize=(10, 10))
for images in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")
    
normalization_layer = tf.keras.layers.Rescaling(1./255)
#%%