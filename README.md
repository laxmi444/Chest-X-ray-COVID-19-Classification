# Chest X-ray COVID-19 Classification using CNN

## Overview
This project aims to classify chest X-ray images to detect COVID-19 using a Convolutional Neural Network (CNN). The model is trained on a dataset of chest X-ray images and can predict whether a given X-ray image indicates a COVID-19 infection or not.

## Model Architecture

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)             (None, 100, 100, 32)      896       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2) (None, 50, 50, 32)        0         
_________________________________________________________________
batch_normalization (BatchNo) (None, 50, 50, 32)        128       
_________________________________________________________________
conv2d_4 (Conv2D)             (None, 50, 50, 64)        18496     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2) (None, 25, 25, 64)        0         
_________________________________________________________________
batch_normalization_1 (Batch) (None, 25, 25, 64)        256       
_________________________________________________________________
conv2d_5 (Conv2D)             (None, 25, 25, 128)       73856     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2) (None, 12, 12, 128)       0         
_________________________________________________________________
batch_normalization_2 (Batch) (None, 12, 12, 128)       512       
_________________________________________________________________
flatten_1 (Flatten)           (None, 18432)             0         
_________________________________________________________________
dense_2 (Dense)               (None, 128)               2359424   
_________________________________________________________________
dropout (Dropout)             (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)               (None, 1)                 129       
=================================================================
Total params: 2,453,697
Trainable params: 2,453,249
Non-trainable params: 448

## Usage
Run the Jupyter Notebook: Open and run the _xray_classification.ipynb notebook to see the complete project implementation.

To run this project, you need to have Python and Jupyter Notebook installed. You can install the necessary dependencies using the following command:

Copy code
pip install numpy pandas tensorflow scikit-learn matplotlib opencv-python

The notebook contains detailed instructions and code cells for:

Copy code
Loading and preprocessing the dataset.
Building, training, and evaluating the CNN model.
Visualizing results and performance metrics.