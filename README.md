# Training an Autonomous Robot Car

## Table of Contents
- [Overview](#overview)
- [Background](#background)
- [Image Data](#image-data)
- [Objectives](#objectives)
- [Model](#model)
    - [Architecture](#architecture)
    - [Construction](#construction)
    - [Implementation](#implementation)
- [Preprocessing](#preprocessing)
- [Training and Optimization](#training-and-optimization)
    - [Loss Functions](#loss-functions)
    - [Learning Rate](#learning-rate)
    - [Validation](#validation)
    - [Regularization](#regularization)
    - [Class Weighting](#class-weighting)
    - [Custom Metrics](#custom-metrics)
    - [Evaluation](#evaluation)
- [Results](#results)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Kaggle Competition](#kaggle-competition)
    - [Live Testing](#live-testing)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [References](#references)
- [License](#license)

## Overview

In this project, a deep multi-task convolutional neural network (CNN) partnered with supervised learning techniques is used to train a small robot car to autonomously navigate various tracks using image data from its onboard camera.

The model is designed and then trained to predict two labels for each image; the speed and the angle the car should immediately employ. Steps are taken to improve model performance and efficiency, and a final version is tested in two phases; a Kaggle competition involving a hidden test set; and live testing involving the implementation of the model on the car and observing its driving proficiency around three tracks. The final model generally performs well at predicting the speed but struggles with predicting the angle in some complex scenarios. However, the model is extremely efficient and can handle driving at significantly increased speeds.

The following sections provide detailed explanations of the most important elements of the project, including details of the task; development of the model; and the final results. There is also a discussion of the results and a conclusion.

Although this was a team-based project, all work posted to this repository is my own.

## Background

Supervised learning is a branch of machine learning defined by its use of labelled data to train models [1]. It involves devising an algorithm allowing a model to learn to predict the correct label for a given sample from the training dataset. The aim is for the trained model to be able to make accurate predictions of labels for unseen data.

CNNs are a type of artificial neural network architecture often used in conjunction with supervised learning. These model designs are defined by their use of convolutional layers that can learn to pick out important spatial features from their inputs by effectively filtering over them [2]. Another critical component of a CNN is the pooling layer, which is typically used immediately after one or more convolutional layers and their activations. Pooling is used to reduce the spatial size of the representation [3]. This is important for allowing the model to learn features at higher levels of abstraction, aiding generality and reducing computational load. Due to their ability to learn complex relationships in high dimensional inputs, CNNs are very well suited for handling image data.

## Image Data

The images used by the model come from the onboard camera of a small electric robot car [4], displayed below. The car also has an onboard Raspberry Pi (RPi) 4 [5] that is used to run the model and make predictions on the image data in real-time.

![The PiCar](figures/car.PNG)

The speed label corresponds to the speed at which the car should drive, given the current image. There are two classes of speed label, 0 or 1, which correspond to the car stopping its wheels or driving them forwards at a constant rate. This means that predicting the speed label is a binary classification problem.

The angle label corresponds to the angle of steering the car should immediately employ, given the current image. There are 17 possible values for the angle label, ranging from 0 to 1, corresponding to the normalised angle of steering, as calculated from the true angle using,

$\rm angle_{norm} = \frac{(angle-50)}{80}$.

When placed in ascending order, the difference in value between consecutive labels corresponds to a 5° difference in angle. This linearity partnered with the continuous nature of angles means that it is reasonable to treat prediction of these angle labels as a linear regression problem.

The inputs of the model are 320x240 RGB PNG images that are taken from a digital camera attached to the front of the car. At each time step, the latest image is sent to the model, and it predicts the speed and angle the car should employ. The idea then, is for the car to autonomously drive around a track based on what it can immediately “see” in front of itself.

The training data to be used to train the model are images taken from the camera whilst the car is expertly remotely driven around the track by a human. For each image captured in this way, the labels are assigned based on the speed and angle of driving that were employed by the human driver at the moment the image was captured.

An initial labelled training dataset is provided. This dataset includes 13,792 labelled images that were previously collected. There is also the option to collect additional data.

Initial training data is stored in "machine-learning-in-science-ii-2023/training_data/combined". Newly collected data is stored in "machine-learning-in-science-ii-2023/training_data/new_data" and is automatically sorted and moved into the former directory prior to training. The test data is stored in "machine-learning-in-science-ii-2023/test_data/test_data". "training_norm.csv" contains the names and corresponding labels of each image in the initially provided training dataset. The data is not uploaded to this repository.

Example image data:

![Example image 1](figures/example1.png)

![Example image 2](figures/example2.png)

![Example image 3](figures/example3.png)

## Objectives

The main objectives of this project are separated into two sections; the Kaggle competition; and the live testing of the car.

The objective of the Kaggle competition is to train a model that achieves a competitively low loss on a hidden labelled test dataset consisting of 1020 images, in competition with other groups. The loss is calculated as the sum of the Mean Squared Error (MSE) of the speed and angle predictions.

In the live testing section, the model is implemented into the physical car and its performance is assessed based on its ability to perform certain tasks. All tasks involve the car driving around a track and reacting to features the model can identify in the images it sees. The three tracks to be used during live testing are displayed below, and include “T-junction”, “Oval” and “Figure-of-eight” tracks.

![The three tracks used to evaluate the model during live testing](figures/tracks.png)

The T-junction track is used to assess the models ability to steer the car along a straight line in the correct lane and to turn either left or right, as indicated by arrows placed at the end of the track.

The Oval track is used to assess the models ability to steer smoothly round corners whilst staying in the correct lane.

The Figure-of-eight track allows evaluation of performance when crossing an intersection. This track is also used to test how the model reacts to traffic lights; it should stop at a red light and continue at a green light.

Additionally, a test is used to evaluate the models ability to stop the car when an obstacle is placed in front of it, and to drive on if the obstacle is not in the road.

## Model

### Architecture

The model must predict two labels; speed and angle. These can be thought of as binary classification and linear regression problems, respectively. The nature of these prediction tasks is clearly very different; the speed prediction is effectively an object detection problem, whereas the angle prediction is simply predicting steering angle. Due to their differences, the predictions for each label are best represented by different loss functions.

The model is therefore designed as a multi-task CNN with two branches resulting in two outputs. The output of one branch is the speed label prediction, with the loss computed as the binary cross-entropy, and the output of the second branch is the angle label prediction, with the loss computed as the MSE.

![An illustration of the final model.](figures/model.png)

The model is illustrated above. Each branch has three convolutional layers involving 32, 64 and 128 3x3 kernels, respectively. Each convolutional layer is followed by a ReLU activation function, to allow the model to learn non-linear relationships. After the first two convolutional layers and their activations, there are max pooling layers, involving 2x2 kernels. Since the input images are RGB, the convolutional and max pooling layers are broken into 3 channels representing the red, green and blue components. After the final convolutional layer and its activation, their output is flattened and passed to two fully connected layers consisting of 64 and 10 neurons. The final output from each branch represents either the speed or angle prediction. With an input image of size 32x32, the total number of trainable parameters of the model is 450,090.

### Construction

The model is constructed in Python [6], making use of the TensorFlow 2 [7] machine learning library. The high-level Keras API [8] is utilized for constructing the layers. The model is structured to be compatible with conversion to TensorFlow Lite [9], TensorFlow’s lightweight solution for mobile and embedded devices. Converting to TensorFlow Lite provides the benefit of low-latency inference when running the model on a TPU. To leverage this benefit, the car is fitted with a Coral Edge TPU [10] connected to the RPi.

The Jupyter notebook "train_model.ipynb" contains the code used to build, train and save the model to a TFLite file.

### Implementation

The TFLite file containing the model is integrated into a script that runs on the RPi called "model.py". This script loads and runs the model and interfaces with the pre-installed software on the car. First, the latest image from the camera is loaded. This is then fed to the model so it can make predictions to determine the cars actions. The outputs from the model are then passed to the software on the car responsible for driving its motors. This sequence of operations is then immediately repeated, creating a continuous cycle of input, prediction, and action.

The folder "PiCar" contains the aforementioned files that are uploaded to the car.

## Preprocessing

Since the raw images taken from the camera are 320x240 and the model requires inputs of size 32x32, the images must initially be resized. This is achieved simply by using the resize method from the tensorflow.image module.

Each image in the training dataset is then attempted to be displayed using the Image module from the Pillow library [11]. If the image cannot be displayed then it is likely corrupted, and is removed from the training set.

Next, image augmentation is applied. This is where alterations and/or distortions are made to an image to give the model an artificially more diverse set of training data to learn from, to combat overfitting. Multiple augmentation techniques are experimented with, including random flip, random rotation, random brightness, random contrast and random saturation (see "augmentations.ipynb" for examples of these augmentations applied to an image from the training dataset). These augmentations are applied in turn and in combination to training images over multiple initial training runs, where their approximate effect on performance may be observed. Random brightness is kept as the augmentation applied in the final model.

## Training and Optimization

### Loss Functions

During training, after all the forward passes of a batch, the loss is backpropagated through the weights of the network. The binary cross entropy loss is used to update the weights of the parameters of the speed branch of the model. The MSE loss is then used equivalently for the angle branch. The approximated gradients of the loss with respect to each weight is used by the Adam optimiser [12] to determine by what degree each weight is updated.

### Learning Rate

To improve training stability, the learning rate parameter of the Adam optimiser is varied during training. Specifically, it exponentially decays from an initial learning rate to a final learning rate over the course of training, in accordance with a learning rate schedule. This ensures that learning is initially fast and then slows over time, allowing fine tuning towards the minimum of the loss function.

### Validation

To assist the training process, the training data is divided into training and validation sets. The validation set is used to evaluate the current models performance after each epoch on a set of images the model has not used for training. A comparison between the training and validation loss offers an estimate of the magnitude of overfitting.

### Regularization

To combat overfitting and to encourage the model to learn more general relationships, regularization techniques are employed and tuned.

Specifically, L2 regularization is applied to each convolutional and fully connected layer. This adds a penalty proportional to the magnitude of each of the weights to the loss function, encouraging the model to minimise each weight [13]. The degree to which L2 regularization is applied for each layer is defined by a set of hyperparameters that must be tuned.

The second technique utilised is dropout. This is defined by a dropout rate that corresponds to the proportion of outputs from a layer that are randomly ignored [14]. Dropout is applied at a rate of 0.5 for all layers of the model.

The hyperparameters of these regularization techniques are tuned through trial and error. First, a set of hyperparameters are selected and the model is trained. The training and validation loss for each epoch are plotted on a graph using the Matplotlib library [15], and the gap between curves is observed. The hyperparameters are then adjusted between trial runs in attempts to minimise this gap.

### Class Weighting

The nature of the binary classification task poses a unique problem. Optimal training is sensitive to the balancing between number of representations of each class in the training dataset. In the training set, there are more examples of the car in motion than not, therefore, there are more images with speed=1 than there are of speed=0. If this misbalancing is adequately large, the model may learn to bias predictions of the majority class, since it is more likely to guess this class correctly. A model that guesses is not desirable; it should instead learn to discriminate between classes.

To encourage learning and to correct for the imbalanced data, class weighting is utilised. This method involves upweighting the minority class during backpropagation. This effectively means that misclassifying a minority class sample will result in a higher loss compared to misclassifying a majority class sample. The value of the weights is set as the ratio between the two classes, and then is fine-tuned through trial and error.

### Custom Metrics

To better observe the models performance with respect to the speed prediction task, custom metrics are defined and tracked throughout training. These include the specificity, Negative Predictive Value (NPV) and F1-negative. These metrics focus on measuring the models effectiveness at predicting the negative class (speed=0). This is because the negative class is the minority class and its samples contain the object the model should learn to detect. Specificity is a metric used to measure the proportion of actual negatives that are correctly identified [16], and is defined as,

$\rm Specificity = \frac{TN}{TN + FP}$,

where TN is the number of true negatives and FP is the number of false positives. NPV is the proportion of predicted negatives that are actual negatives, and is defined as,

$\rm NPV = \frac{TN}{TN + FN}$,

where FN is the number of false negatives. F1-negative is the equivalent of the standard F1 score [17] but for the negative class. This combines the specificity and NPV into a single metric that can be used as a general indicator of predictive performance of the negative class. It is defined as,

$\rm F1_{negative} = \frac{NPV * Specificity}{NPV + Specificity}$.

### Evaluation

A proportion of the initial training data is also split into an evaluation set. This is a dataset separate from the training and validation set that is never seen by the model and is balanced in numbers of samples from each class of speed label. The model makes predictions on the evaluation set after each epoch and the custom metrics are calculated. The values of these metrics on the evaluation set give a clear indication of predictive success at the classification task, since if the model is guessing, the specificity and NPV will be either 0 or 1. A value in between this range suggests the model is attempting to learn to classify, an indication that the class weighting parameters are set correctly.

## Results

### Evaluation Metrics

The final models learning curve on the evaluation set over training on 200 epochs with a batch size of 40 is shown below. The loss is separated into its speed and angle components and is measured as the MSE. The model seems to settle quickly on an optimal angle loss but the speed loss seems unstable although generally decreasing.

![The MSE of predictions on the evaluation set over 200 training epochs.](figures/loss.png)

The values of the specificity and F1-negative metrics associated with the evaluation set throughout training are shown below. Both metrics seem to be approaching 1, indicating successful learning of classification between the two classes.

![The F1-negative and specificity of speed predictions on the evaluation set over 200 training epochs.](figures/speed.png)

### Kaggle Competition

For the Kaggle competition, an earlier model was used which achieved a combined loss of 0.03758, a competitive score in the desirable range between 0.01-0.06.

### Live Testing

During live testing, the car was excellent at stopping before colliding with obstacles and ignoring them on the side of the road, but failed to stop at red lights, indicating good overall performance at the classification task. The car successfully navigated the oval track and could handle driving at a significantly increased speed. However, it failed to drive in a straight line down the T-junction track or turn at the junction. It also failed to cross the intersection of the Figure-of-eight track, although handling the turns.

The following are videos of the car driving autonomously using the trained model.

The car navigating the inside lane turn of the oval track (early model):

https://github.com/harryd337/Autonomous_PiCar/assets/100152207/f5dd4e58-5fc4-43f2-9d13-9218194fd208

An example of the models ability to recognise obstacles:

https://github.com/harryd337/Autonomous_PiCar/assets/100152207/4ca75cc2-35e0-45de-95c8-cf6628d39027

The car successfully handling a corner at maximum speed:

https://github.com/harryd337/Autonomous_PiCar/assets/100152207/e696aa6b-0c03-4166-bb3f-82aa439a5d06

## Discussion

An initial problem was deciding which model design to focus on. Initially, transfer learning was attempted using the MobileNetV2 pretrained network [18]. All parameters were frozen during training except for the output weights. Performance was okay using this method, but training time and computational load were greatly increased. Earlier layers could not be retrained due to GPU memory limitations. Using such a large model would also increase inference time, of which minimisation was a high priority.

The final model was settled on due to its lightweight design, minimal trainable parameters, small input dimensions and an ability for customization. A small model gives the benefit of faster inference time, especially when utilizing the TPU. This benefit was displayed by the cars exceptional performance when driving speed was increased to the maximum on the oval track and the model successfully kept up with the fast-changing environment. However, likely due to its simplicity, the model struggled on many of the most difficult tasks during live testing. The model clearly struggled with learning complex scenarios such as crossing the intersection and turning at the T-junction. A larger model with more layers handling larger inputs may be better suited for representing this complexity.

The model could also be redesigned to better make use of its branched structure. It is likely that very similar representations were learned in both branches on the earlier layers. Removing one of these likely duplicate layers would free up compute that could be spent on learning in additional deeper layers, perhaps improving performance. Another possibility to explore is with sharing information between branches for certain layers. This could help spread learning over both branches and increase generality of a larger model.

An important limitation to performance on the most difficult tasks that is theorised is the lack of sufficient data that represents these scenarios. In testing prototype models, these scenarios were identified and a total of 9,803 additional images were consequently collected. Sparse images of minimally represented situations, including the first frame as the car approaches the intersection, were duplicated to upweight their importance. This additional data certainly improved the models abilities but it is firmly believed that far more data is required to improve it further.

A key mistake was removing all traffic light data, as this was originally thought to be confusing the model. This is clearly the reason for the final model being totally incapable of comprehending traffic lights.

## Conclusions

In conclusion, the model was designed and trained well enough to achieve good results in both phases of testing. Except for with traffic lights, the model saw great success at the classification task, and could stop before colliding with obstacles. For the regression task, the model performed well on the curves of the tracks but failed at some of the more difficult tasks, such as crossing the intersection and turning at the T-junction. However, the model was extremely efficient with very low inference times, allowing the car to drive at increased speeds. It is theorised that performance could be improved by acquiring more data and redesigning the model to better handle the complexity of the most difficult tasks.

## References

[1] C. M. Bishop and N. M. Nasrabadi, Pattern recognition and machine learning, J. Electronic Imaging 16, 049901 (2006).

[2] S. Albawi, T. A. Mohammed, and S. Al-Zawi, Understanding of a convolutional neural network, in [2017 International Conference on Engineering and Technology (ICET)](https://doi.org/10.1109/ICEngTechnol.2017.8308186) (2017) pp. 1–6.

[3] Qualcomm Technologies, Inc., [Deep learning and convolutional neural networks for computer vision](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/learning-resources/cnn-architectures/deep-learning-convolutional-neural-networks-computer-vision) (2023), accessed: 28/05/2023.

[4] SunFounder, [Sunfounder picar-v kit v2](https://www.sunfounder.com/products/smart-video-car) (2020), accessed: 02/06/2023.

[5] Raspberrypi.org, [Raspberry pi 4](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) (2020), accessed: 02/06/2023.

[6] Python Software Foundation, [Python language reference, version 3.9](https://docs.python.org/3.9/) (2020), available at python.org.

[7] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man´e, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Vi´egas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, [TensorFlow: Large-scale machine learning on heterogeneous systems](https://www.tensorflow.org/) (2015), software available from tensorflow.org.

[8] F. Chollet et al., Keras, [keras.io](https://keras.io/) (2015).

[9] G. Inc., Tensorflow lite, [https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) (2017).

[10] G. LLC., [Tensorflow models on the edge tpu](https://coral.ai/docs/edgetpu/models-intro/) (2020), accessed: 02/06/2023.

[11] J. A. C. (Alex) and contributors, Pillow, [https://github.com/python-pillow/Pillow](https://github.com/python-pillow/Pillow) (2023).

[12] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization (2017), [arXiv:1412.6980 [cs.LG]](https://arxiv.org/abs/1412.6980).

[13] C. Cortes, M. Mohri, and A. Rostamizadeh, L2 regularization for learning kernels (2012), [arXiv:1205.2653 [cs.LG]](https://arxiv.org/abs/1205.2653).

[14] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, Dropout: A simple way to prevent neural networks from overfitting, [Journal of Machine Learning Research 15, 1929 (2014)](http://jmlr.org/papers/v15/srivastava14a.html).

[15] J. D. Hunter, Matplotlib: A 2d graphics environment, [Computing in Science & Engineering 9, 90 (2007)](https://doi.org/10.1109/MCSE.2007.55).

[16] R. Trevethan, Sensitivity, specificity, and predictive values: Foundations, pliabilities, and pitfalls in research and practice, [Frontiers in public health 5, 307 (2017)](https://doi.org/10.3389/fpubh.2017.00307).

[17] D. J. Hand, Assessing the performance of classification methods, [International Statistical Review 80, 400 (2012)](https://doi.org/https://doi.org/10.1111/j.1751-5823.2012.00183.x), [https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1751-5823.2012.00183.x](https://arxiv.org/abs/https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1751-5823.2012.00183.x).

[18] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.C. Chen, Mobilenetv2: Inverted residuals and linear bottlenecks, in [2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition](https://doi.org/10.1109/CVPR.2018.00474) (2018) pp. 4510–4520

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
