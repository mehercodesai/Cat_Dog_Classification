# Cat vs. Dog Image Classification using Transfer Learning (ResNet50)
> Note: This is a practice project designed to explore and demonstrate computer vision concepts, data preprocessing pipelines, and transfer learning workflows using deep learning.

A deep learning project implementing Transfer Learning with ResNet50 and TensorFlow/Keras to classify images of cats and dogs. The pipeline is built to run seamlessly in Google Colab, leveraging pre-trained ImageNet weights for rapid and accurate feature extraction.

## Project Overview
Building an image classifier from scratch requires massive datasets and substantial compute power. As a hands-on learning exercise, this project utilizes Transfer Learning, freezing the convolutional base of a pre-trained ResNet50 model and appending a custom classification head to categorize pet images with high efficiency.   

## Tech Stack & Libraries
- Python 3.x
- TensorFlow / Keras (Deep Learning Framework)
- OpenCV (cv2) (Image Processing & Color Conversion)
- Scikit-Learn (Data Splitting & Metrics)
- Seaborn & Matplotlib (Data Visualization)
- Pandas & NumPy (Data Manipulation)
  
## Project Pipeline & ArchitectureData 
### 1. Acquisition & Setup:
Authenticates with the Kaggle API to download and extract the dataset (dogs-vs-cats-redux-kernels-edition).
### 2. Preprocessing & Augmentation:
- Walks through directories to gather image paths.
- Resizes images to standard dimensions ($224 \times 224$).
- Converts BGR images (OpenCV default) to RGB.
- Applies ResNet50 specific input preprocessing (preprocess_input).
- Encodes labels using one-hot encoding ([1, 0] for cats, [0, 1] for dogs).
### 3. Exploratory Data Analysis (EDA): 
Visualizes class balance and displays grid layouts of training/testing images using matplotlib and seaborn.   
### 4. Model Building (Transfer Learning):
- Base Model: ResNet50 loaded with pre-trained imagenet weights (include_top=False, pooling='max').
- Freezing Layers: Set model.layers[0].trainable = False to preserve learned feature extractors.
- Custom Classifier Head: Added a Dense output layer with softmax activation for 2-class probability distribution.
### 5. Compilation: 
Compiled using categorical cross-entropy loss and optimized via SGD/Adam

## Model SummaryBase 
- Architecture: ResNet50 (Convolutional Feature Extractor)
- Total Parameters: ~23.59 Million
- Trainable Parameters: Minimal (Dense classification head)
- Non-trainable Parameters: ~23.58 Million (Frozen ResNet weights)
