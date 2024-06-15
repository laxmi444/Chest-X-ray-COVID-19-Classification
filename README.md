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
 ```

 # Usage 

 Run the Jupyter Notebook: Open and run the "Chest Xray Classification using CNN.ipynb" notebook to see the complete project implementation.
```bash
jupyter notebook Chest Xray Classification using CNN.ipynb
```

### Run the Streamlit App

```bash
streamlit run app.py
```
### Insert Normal or COVID X-ray Images
To use the model, insert chest X-ray images into the Streamlit app (app.py). The model will predict whether the image indicates a normal condition or COVID-19 infection.

The notebook contains detailed instructions and code cells for:

1. Loading and preprocessing the dataset.
2. Building, training, and evaluating the CNN model.
3. Visualizing results and performance metrics.