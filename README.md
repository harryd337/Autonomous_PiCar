# Training an Autonomous Robot Car

## Overview

In this project, a deep convolutional neural network partnered with supervised learning techniques is used to train a small robot car to autonomously navigate various tracks using image data from its onboard camera. The model is designed and then trained to predict two labels for each image; the speed and the angle the car should immediately employ. Steps are taken to improve model performance and efficiency, and a final version is tested in two phases; a Kaggle competition involving a hidden test set; and live testing involving the implementation of the model on the car and observing its driving proficiency around three tracks. The final model generally performs well at predicting the speed but struggles with predicting the angle in some complex scenarios. However, the model is extremely efficient and can handle driving at significantly increased speeds.

## Image Data

The images come from the onboard camera of a small electric robot car[4]. The car also has an onboard Raspberry Pi (RPi) 4[5] that is used to run the model and make predictions on the image data in real-time.

The speed label corresponds to the speed at which the car should drive, given the current image. There are two classes of speed label, 0 or 1, which correspond to the car stopping its wheels or driving them forwards at a constant rate. This means that predicting the speed label is a binary classification problem.

The angle label corresponds to the angle of steering the car should immediately employ, given the current image. There are 17 possible values for the angle label, ranging from 0 to 1, corresponding to the normalised angle of steering, as calculated from the true angle using,

$\rm angle_{norm} = \frac{(angle-50)}{80}$.

When placed in ascending order, the difference in value between consecutive labels corresponds to a 5° difference in angle. This linearity partnered with the continuous nature of angles means that it is reasonable to treat prediction of these angle labels as a linear regression problem.

The inputs of the model are 320x240 RGB PNG images that are taken from a digital camera attached to the front of the car. At each time step, the latest image is sent to the model, and it predicts the speed and angle the car should employ. The idea then, is for the car to autonomously drive around a track based on what it can immediately “see” in front of itself.

The training data to be used to train the model are images taken from the camera whilst the car is expertly remotely driven around the track by a human. For each image captured in this way, the labels are assigned based on the speed and angle of driving that were employed by the human driver at the moment the image was captured.

An initial labelled training dataset is provided. This dataset includes 13,792 labelled images that were previously collected. There is also the option to collect additional data.

## Objectives

The main objectives of this project are separated into two sections; the Kaggle competition; and the live testing of the car.

The objective of the Kaggle competition is to train a model that achieves a competitively low loss on a hidden labelled test dataset consisting of 1020 images, in competition with other groups. The loss is calculated as the sum of the Mean Squared Error (MSE) of the speed and angle predictions.

In the live testing section, the model is implemented into the physical car and its performance is assessed based on its ability to perform certain tasks. All tasks involve the car driving around a track and reacting to features the model can identify in the images it sees. The three tracks to be used during live testing are displayed in Figure 1, and include “T-junction”, “Oval” and “Figure-of-eight” tracks.

Figure 1

The T-junction track is used to assess the models ability to steer the car along a straight line in the correct lane and to turn either left or right, as indicated by arrows placed at the end of the track.

The Oval track is used to assess the models ability to steer smoothly round corners whilst staying in the correct lane.

The Figure-of-eight track allows evaluation of performance when crossing an intersection. This track is also used to test how the model reacts to traffic lights; it should stop at a red light and continue at a green light.

Additionally, a test is used to evaluate the models ability to stop the car when an obstacle is placed in front of it, and to drive on if the obstacle is not in the road.

## Model Architecture

The model must predict two labels; speed and angle. These can be thought of as binary classification and linear regression problems, respectively. The nature of these prediction tasks is clearly very different; the speed prediction is effectively an object detection problem, whereas the angle prediction is simply predicting steering angle. Due to their differences, the predictions for each label are best represented by different loss functions.

The model is therefore designed as a multi-output CNN with two branches resulting in two outputs. The output of one branch is the speed label prediction, with the loss computed as the binary cross-entropy, and the output of the second branch is the angle label prediction, with the loss computed as the MSE.

Figure 2

The model is illustrated in Figure 2. Each branch has three convolutional layers involving 32, 64 and 128 3x3 kernels, respectively. Each convolutional layer is followed by a ReLU activation function, to allow the model to learn non-linear relationships. After the first two convolutional layers and their activations, there are max pooling layers, involving 2x2 kernels. Since the input images are RGB, the convolutional and max pooling layers are broken into 3 channels representing the red, green and blue components. After the final convolutional layer and its activation, their output is flattened and passed to two fully connected layers consisting of 64 and 10 neurons. The final output from each branch represents either the speed or angle prediction. With an input image of size 32x32, the total number of trainable parameters of the model is 450,090.
