

# Machine Learning Assignment 2

This repository contains various machine learning assignments focused on Gradient Descent, K-Means Clustering, Support Vector Machines (SVM), and Perceptron algorithms. Each notebook aims to provide an in-depth understanding of these algorithms along with their practical implementations.

## Table of Contents

1. [Gradient Descent and Convex Functions](#1-gradient-descent-and-convex-functions)
2. [K-Means and Kernelized Clustering](#2-k-means-and-kernelized-clustering)
3. [Support Vector Machine (SVM)](#3-support-vector-machine-svm)
4. [Perceptron Algorithm](#4-perceptron-algorithm)
5. [HTRU2 Pulsar Dataset](#5-htru2-pulsar-dataset)

## 1. Gradient Descent and Convex Functions

**Notebook:** [Harsh_Shelke_GD.ipynb](https://github.com/harshelke180502/ML_ASSIGNMENT_2/blob/main/Harsh_Shelke_GD.ipynb)

### Objective
In this notebook, you will explore Gradient Descent (GD) and Projected Gradient Descent (PGD) on convex functions. The key objectives include:
- Understanding the concept of convex functions.
- Implementing Gradient Descent (GD) for optimization.
- Visualizing the convergence rate of GD over iterations.
- Implementing Projected Gradient Descent (PGD) for constrained optimization.
- Comparing the performance of GD vs. PGD.

### Contents
1. **Load Libraries and Define Convex Function**: Import necessary libraries and define convex functions.
2. **Implementation of Gradient Descent (GD)**: Implement and visualize the Gradient Descent algorithm.
3. **Implementation of Projected Gradient Descent (PGD)**: Implement and visualize the Projected Gradient Descent algorithm.
4. **Questions and Analysis**: Answer conceptual questions related to GD and PGD.

## 2. K-Means and Kernelized Clustering

**Notebook:** [Harsh_Shelke_NN.ipynb](https://github.com/harshelke180502/ML_ASSIGNMENT_2/blob/main/Harsh_Shelke_NN.ipynb)

### Objective
In this assignment, you will:
- Implement the K-Means clustering algorithm from scratch.
- Apply K-Means to cluster non-linearly separable data.
- Understand the limitations of K-Means and motivate the need for kernelization.
- Implement a basic form of Kernel K-Means using the RBF kernel.

### Contents
1. **Load Libraries and Dataset**: Import necessary libraries and load the dataset.
2. **Generate and Visualize a Non-Linearly Separable Dataset**: Use sklearn's `make_moons` to generate 2D data.
3. **Implement K-Means Algorithm**: Implement the K-Means clustering algorithm and visualize the results.
4. **Kernel K-Means**: Implement Kernel K-Means using the RBF kernel and compare the performance.

## 3. Support Vector Machine (SVM)

**Notebook:** [Harsh_Shelke_SVM.ipynb](https://github.com/harshelke180502/ML_ASSIGNMENT_2/blob/main/Harsh_Shelke_SVM.ipynb)

### Objective
Work with the Predicting a Pulsar Star dataset and use Support Vector Machines (SVM) to classify pulsar stars. The objectives include:
- Performing Exploratory Data Analysis (EDA).
- Preprocessing the dataset (handling missing values, outliers, and scaling).
- Implementing Soft-Margin SVM and Kernel SVM.
- Tuning hyperparameters (C and kernel type) to optimize model performance.
- Comparing different models and interpreting results.

### Contents
1. **Load Libraries and Dataset**: Import necessary libraries and load the HTRU2 Pulsar dataset.
2. **Exploratory Data Analysis (EDA)**: Perform EDA to understand the dataset.
3. **Data Preprocessing**: Handle missing values, outliers, and scale the features.
4. **Implement Soft-Margin SVM**: Implement and train a Soft-Margin SVM model.
5. **Kernel SVM**: Implement and train a Kernel SVM model.
6. **Hyperparameter Tuning**: Tune hyperparameters to optimize the model performance.
7. **Comparison and Results**: Compare different models and interpret the results.

## 4. Perceptron Algorithm

**Notebook:** [Harsh_Shelke_perceptron.ipynb](https://github.com/harshelke180502/ML_ASSIGNMENT_2/blob/main/Harsh_Shelke_perceptron.ipynb)

### Objective
In this exercise, you will implement the Perceptron learning algorithm on a dataset generated from a linearly separable logistic regression model.

### Contents
1. **Data Generation**: Generate a synthetic dataset of 100 two-dimensional points.
2. **Implement the Perceptron Algorithm**: Implement the Perceptron update rule for online learning.
3. **Visualization**: Visualize the data points and the decision boundary at different stages of training.
4. **Kernel Perceptron**: Implement a kernelized perceptron to classify data that is not linearly separable.

## 5. HTRU2 Pulsar Dataset

**File:** [htru2.zip](https://github.com/harshelke180502/ML_ASSIGNMENT_2/blob/main/htru2.zip)

### Description
The HTRU2 dataset contains pulsar candidates collected during the High Time Resolution Universe Survey. The dataset is used in the SVM assignment to classify pulsar stars.

### Contents
- **HTRU_2.csv**: The CSV file containing the dataset.

## Getting Started

To get started with the notebooks, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/harshelke180502/ML_ASSIGNMENT_2.git
