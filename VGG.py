#calling the VGG model and preparing the data 
from tensorflow.keras.applications.vgg16 import VGG16 #calling the model 
from tensorflow.keras.applications.vgg16 import preprocess_input #calling the preprocessing unit

#Loading the VGG model 
standard_model = VGG16(weights="imagenet",include_top=False, input_shape=train_data[0].shape)
standard_model.trainable=FALSE #freeze the model

#preprocessing unit applied on the training and validation datasets
training_data = preprocess_input(training_data) #replace training_data with whatever the training set is called in your alogrithm
validation_data = preprocess_input(validation_data) #same for the validation set

standard_model.summary()


#building the model and adding the last layer
flatten_layer = layers.Flatten()
#the VGG model ends with a maxpooling layer so maybe adding dense layers at the end would be a good idea
#dense_layer_1 = layers.Dense(50, activation='relu')
#dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(5, activation = 'softmax')

model = model.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])   #depending on how the model is built dense_layer_1 and 2 may be excluded

#compiling the model

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

history = model.fit(train_batches,
                   epochs=EPOCHS,
                   validation_data=validation_batches)  #replace validation batches
