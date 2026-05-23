# Spam Email Classifier using Decision Tree

This project builds a machine learning model to classify emails as **Spam** or **Not Spam** using the Spambase dataset.

## Project Overview

The goal of this project is to apply a full machine learning workflow, including:

- Splitting the dataset into training, validation, and testing sets
- Comparing model performance before and after normalization
- Detecting and fixing overfitting using tree pruning
- Handling class imbalance using SMOTE
- Adjusting the classification threshold
- Evaluating the final model on unseen test data
- Visualizing the decision tree and extracting decision rules

## Dataset

The project uses the Spambase dataset, which contains numerical features related to word frequencies, character frequencies, and capital letter patterns in emails.

Target column:

- `0` = Not Spam
- `1` = Spam

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Decision Tree Classifier
- SMOTE

## Methodology

### 1. Data Splitting

The dataset was split into:

- 70% training set
- 15% validation set
- 15% testing set

### 2. Normalization Comparison

Min-Max normalization was applied and compared against the original data.  
The results showed that normalization did not affect the Decision Tree performance because Decision Trees are scale-invariant.

### 3. Overfitting Analysis

Training and validation accuracy were compared across different tree depths.  
The model showed signs of overfitting as tree depth increased.

### 4. Overfitting Fix

The Decision Tree was pruned using:

```python
max_depth = 5
