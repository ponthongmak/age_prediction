# <b> Age Prediction from face image by Deep Learning
  
## <b>Introduction
  
<p style="text-indent: 2.5em;">
    Age and gender are the key facial attributes, play an essential role in social interactions. Automatic age and gender prediction have become relevant to an increasing amount of applications, particularly when social platforms and social media become common in human's daily life. Estimation of age and gender from a face image is a critical task in artificial intelligence applications, such as access control, human-computer interaction, law enforcement, marketing intelligence, and visual surveillance.
    
<p style="text-indent: 2.5em;">    
    In deep learning, the Convolutional Neural Network (CNNs) is one of the primary methods to do images recognition, images classifications. Objects detections, recognition faces, etc., are some of the areas where CNNs are widely used. CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons is a fully connected network; that is, each neuron in one layer is connected to all neurons in the next layer. The multilayer perceptrons always prone to overfitting data. However, CNNs take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns to regularize to reduce overfitting.
    
<p style="text-indent: 2.5em;">
    The aim of this study is to create a model to classify age group from the Adience face image project, the face images provided by the open university of Israel. The aligned face images are retrieved, then resize and feed into the model in order to train the best model for face prediction.
  
## <b>Data Acquisition
<p style="text-indent: 2.5em;">
  The data used in this study derived from Adience face image project, the open university of Israel. The data is downloaded then extract into the computer drive. The link of the dataset is https://talhassner.github.io/home/projects/Adience/Adience-data.html . The data used in this study include, fold.txt files and image files in the aligned folder which derived from extract aligned.tar.gz as shown in the figure below,


## Required Libraries
from numpy.random import seed

from tensorflow import set_random_seed

import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import cv2

import keras

from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten,\
ZeroPadding2D, Convolution2D, Activation, GlobalMaxPooling2D

from keras.applications.resnet50 import ResNet50

from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve,\
average_precision_score, f1_score, auc, balanced_accuracy_score 

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

from keras.utils import to_categorical

from itertools import cycle

from collections import Counter

from scipy import interp

import itertools

### <b> Conclusions
<p style="text-indent: 2.5em;">
    The study aims to predict age from face images, the model using deep learning, the convolutional neural network with transfer learning. The pre-trained networks used in this study are Resnet50 (with imagenet weight), and VGG faces (with VGG face weight). The aligned face images are retrieved from the Adience face image project, the open university of Israel, then resize and feed into the model in order to train the best model for face prediction.
<p style="text-indent: 2.5em;">
    The VGG face architecture (with VGG face weight) performs better than Resnet50 architecture (with imagenet weight). As a result, the comprehensive evaluation only performed on VGG model. There are eight classes of age including 0-2, 4-6, 8-12, 15-20, 25-32, 27-43, 48-53, and 60-100. Due to imbalanced class, the evaluation matric used F1, the harmonic mean of precision and recall, and another evaluation matric is balanced accuracy, which is a harmonic mean of sensitivity and specificity. The model F1 is 0.84, and the balanced accuracy is 0.80 The model is fit well with image data and more likely to generalize when applying with another dataset.
