# Plant Disease Classification using Support Vector Machines (SVM)

This project focuses on classifying plant diseases using images of plant leaves. By employing Support Vector Machines (SVM), this model aims to effectively differentiate between healthy plants and those affected by various diseases.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The primary objective of this project is to develop an image classification model that can accurately identify the health status of plants based on leaf images. The model uses a linear SVM algorithm, which is trained on labeled data to classify images into categories: healthy, multiple diseases, rust, and scab.

## Dataset

The dataset used in this project is the **Plant Pathology 2020** dataset, which includes:
- `train.csv`: Contains the training labels for images, including classes for healthy plants and various diseases.
- `test.csv`: Contains images for which predictions need to be made.
- `images/`: Directory containing all the images.

## Installation

To set up the project, you need to have the following libraries installed:

```bash
pip install pandas numpy opencv-python scikit-learn
Usage
Load the Dataset: Update the paths in the script to point to your local dataset.
Run the Script: Execute the script to train the SVM model and generate predictions for the test dataset.
bash
Copy code
python plant_disease_classification.py
Output: The predicted labels for the test dataset will be saved to test_with_predictions.csv.
Model Training
The dataset is split into training and validation sets to evaluate model performance.
Features are extracted from images by resizing and flattening them to a uniform size (128x128 pixels).
The SVM model is trained on the training set and evaluated on the validation set, providing metrics such as accuracy, precision, recall, and F1-score.
Key Steps:
Feature Extraction: Each image is converted to grayscale and resized, and the features are flattened for input into the model.
Model Training: The SVM model is trained using the extracted features and their corresponding labels from the training data.
Evaluation: The model's performance is assessed on the validation set, generating a classification report and accuracy score.
Results
The model outputs predictions for the test dataset, which are saved to a CSV file for further analysis. Detailed classification reports can also be generated to assess model performance, including metrics such as precision, recall, and F1-score.
