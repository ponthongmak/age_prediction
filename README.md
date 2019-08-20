# Age Prediction from face image by Deep Learning
  
## Introduction
    Age and gender are the key facial attributes, play an essential role in social interactions. Automatic age and gender prediction have become relevant to an increasing amount of applications, particularly when social platforms and social media become common in human's daily life. Estimation of age and gender from a face image is a critical task in artificial intelligence applications, such as access control, human-computer interaction, law enforcement, marketing intelligence, and visual surveillance.
    
    In deep learning, the Convolutional Neural Network (CNNs) is one of the primary methods to do images recognition, images classifications. Objects detections, recognition faces, etc., are some of the areas where CNNs are widely used. CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons is a fully connected network; that is, each neuron in one layer is connected to all neurons in the next layer. The multilayer perceptrons always prone to overfitting data. However, CNNs take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns to regularize to reduce overfitting.
   
    The aim of this study is to create a model to classify age group from the Adience face image project, the face images provided by the open university of Israel. The aligned face images are retrieved, then resize and feed into the model in order to train the best model for face prediction.
  
## Data Acquisition
  The data used in this study derived from Adience face image project, the open university of Israel. The data is downloaded then extract into the computer drive. The data used in this study include, fold.txt files and image files in the aligned folder which derived from extract aligned.tar.gz.
  
Download Dataset:> http://www.openu.ac.il/home/hassner/Adience/data.html

## Required Libraries
- Python 3.6.8
- Keras 2.2.4
- OpenCV 4.1.0
- Sklearn 0.21.3
- joblib 0.13.2

## Pre-trained model (for transfer learning)
    - VGG-Very-Deep-16 CNN
    - Resnet50
    
## Model wieght training
    - VGGFace
    - Imagenet

## Conclusions
    The study aims to predict age from face images, the model using deep learning, the convolutional neural network with transfer learning. The pre-trained networks used in this study are Resnet50 (with imagenet weight), and VGG faces (with VGG face weight). The aligned face images are retrieved from the Adience face image project, the open university of Israel, then resize and feed into the model in order to train the best model for face prediction.

    The VGG face architecture (with VGG face weight) performs better than Resnet50 architecture (with imagenet weight). As a result, the comprehensive evaluation only performed on VGG model. There are eight classes of age including 0-2, 4-6, 8-12, 15-20, 25-32, 27-43, 48-53, and 60-100. Due to imbalanced class, the evaluation matric used F1, the harmonic mean of precision and recall, and another evaluation matric is balanced accuracy, which is a harmonic mean of sensitivity and specificity. The model F1 is 0.84, and the balanced accuracy is 0.80 The model is fit well with image data and more likely to generalize when applying with another dataset.
