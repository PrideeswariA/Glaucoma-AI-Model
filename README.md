# Glaucoma Detection Using MobileNetV2

## Project Overview
This project uses deep learning and transfer learning with MobileNetV2
to classify retinal fundus images into Glaucoma and Normal categories.

## Objective
The objective is to develop an image classification model that can
identify glaucoma-related patterns in retinal fundus images.

## Dataset
Dataset: Cataract, DR, Normal and Glaucoma Fundus Images Dataset from Kaggle.

For this project, only two classes are used:
- Glaucoma
- Normal

Dataset Link:
https://www.kaggle.com/datasets/drskprabhakar/cataract-dr-normal-glaucoma-fundus-images-dataset

## Model Architecture
- MobileNetV2 pretrained on ImageNet
- Global Average Pooling
- Dropout
- Sigmoid output layer
- Adam optimizer
- Binary Cross-Entropy loss

## Dataset Split
- Training images: 1455
- Validation images: 416
- Testing images: 210

## Model Performance
- Test Accuracy: 90.95%
- Test Loss: 0.2140

## Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Repository Files
- `glaucoma_model.ipynb` - Model training and evaluation notebook.
- `mobilenetv2_glaucoma_model.keras` - Trained model file.

## Disclaimer
This project is developed for educational and research purposes only.
It is not intended to replace professional medical diagnosis.
