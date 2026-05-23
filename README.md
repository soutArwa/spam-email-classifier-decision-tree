# Spam Email Classification

A machine learning project focused on classifying emails as **Spam** or **Not Spam** using a Decision Tree classifier.

This project demonstrates a complete supervised machine learning workflow, from dataset preparation and model training to overfitting analysis, class imbalance handling, threshold tuning, and final evaluation.

---

## Overview

Spam detection is a common classification problem in machine learning where the goal is to identify whether an email is unwanted spam or a legitimate message.

In this project, a Decision Tree model was built to classify emails based on numerical patterns extracted from email content, including word frequencies, character frequencies, and capital letter usage.

The workflow was designed to go beyond simple model training by analyzing model behavior, improving generalization, and addressing real-world classification challenges such as imbalanced data.

---

## Dataset

This project uses the **Spambase dataset**, created by Hopkins et al. in 1999.

The dataset contains **4,601 email instances**, with **57 numerical attributes** and one binary class label.

| Email Type | Class Label | Instances | Percentage |
|---|---:|---:|---:|
| Spam | 1 | 1,813 | 39.4% |
| Not Spam | 0 | 2,788 | 60.6% |
| Total | - | 4,601 | 100% |

Each email is represented using numerical features that describe the frequency of specific words, characters, and capital letter sequences.

---

## Project Objectives

- Build a Decision Tree classifier for spam email detection
- Split the dataset into training, validation, and testing sets
- Compare model behavior before and after normalization
- Analyze overfitting using training and validation curves
- Improve generalization through tree pruning
- Handle class imbalance using SMOTE
- Adjust the decision threshold to improve spam detection
- Evaluate the final model on unseen test data
- Visualize the Decision Tree and extract readable decision rules

---

## Machine Learning Workflow

### 1. Data Preparation

The dataset was divided into three parts:

| Dataset Split | Percentage |
|---|---:|
| Training Set | 70% |
| Validation Set | 15% |
| Testing Set | 15% |

The training set was used to train the model, the validation set was used for tuning and comparison, and the test set was kept unseen until final evaluation.

---

### 2. Normalization Analysis

Min-Max normalization was applied to compare Decision Tree performance before and after scaling.

The results showed that normalization did not significantly change the model’s predictions. This is expected because Decision Trees are scale-invariant models that split data based on thresholds rather than distance-based calculations.

---

### 3. Overfitting Detection

The model was evaluated across different tree depths by comparing training accuracy and validation accuracy.

As the tree depth increased, training accuracy continued to improve while validation accuracy eventually stopped improving. This indicated that the model was beginning to memorize the training data instead of generalizing well.

---

### 4. Model Pruning

To reduce overfitting, the Decision Tree was pruned by limiting its maximum depth.

```python
max_depth = 5
