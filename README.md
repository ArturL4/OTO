# OTO (Online Training Optimazation)

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Description](#description)
4. [Results](#results)

## Installation <a name= "installation"></a>
To use OTO the standard libraries given with the Anaconda Distribution are necessary.<br>
Furthermore you need to install the PyTorch Framework as described [here](https://pytorch.org/get-started/locally/).


## Project Motivation <a name="motivation"></a>

Training a Neural Network with static Hyperparameter Configurations can be very time consuming and might don't even result in good accuracies.<br>
The idea behind OTO is to actually give the Network itself the oppurtinity to decide when to change the Hyperparameters.<br>
We believe that giving the Network a variety of Hyperparameters while training it is possible to reduce the error and enhance the training time.

## Description <a name="description"></a>

To get a benchmark we decided to build a very simple CNN Architecture and the Dataset we were using is the already preprocessed CIFAR10 coming together with the PyTorch Framework. Simply because it is less time consuming and the results can still be compared! <br>
We wanted to compare our method with the static Hyperparameter Configurations you get with Grid- and Random Search.<br>
Learning Rate is the only Hyperparameter we wanted to change dynamically since it is possible to expand the idea if the results seem good.<br>

We split the Dataset into Training and Validation Data and wrote a simple loop to train different models with Grid and Random Search. For comparison we saved the results of Training Accuracy, Validation Accuracy and the error into a csv file.<br>

After getting the results for Grid and Random the trainingsloop has been expanded. The Network got a list of different Learning Rates between 0.1 and 0.0001.<br>
With this pool of different Learning Rates the same amount of Models were trained for 2 Epochs. After that the different models are validated. Picking the best one as new starting point it is trained again with all the different Learning Rates, deciding again which is the best model after 2 Epochs and so on... Iterating to the best result.


## Results <a name="results"></a>
Comparing the different methods we can say that we got a better accuracy with OTO. But not only better results with the Validation we also were able to reduce overfitting which is probably because the Network is validated every two epochs and the Learning Rate can be corrected in time.

