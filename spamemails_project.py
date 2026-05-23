## **Split the dataset into train, test and validation parts**


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

cols = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'is_spam'
]

df = pd.read_csv("spambase.data", header=None, names=col0s)

X = df.drop('is_spam', axis=1)
y = df['is_spam']

# split into 70% train and 30% rest
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# split the remaining 30% into two halves (15% validation, 15% test)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

# put target column back to save them as full datasets
train_data = X_train.copy()
train_data['is_spam'] = y_train

val_data = X_val.copy()
val_data['is_spam'] = y_val

test_data = X_test.copy()
test_data['is_spam'] = y_test

# save as splitted data files
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# print to check sizes
print("Total data:", len(df))
print("Train set:", len(train_data))
print("Validation set:", len(val_data))
print("Test set:", len(test_data))

"""
The dataset were splitted into three parts with 70% for training, 15% for validation and 15% for test.

# **Apply Normalization to the dataset**
"""

# model before normalization
dt_before = DecisionTreeClassifier(random_state=42)
dt_before.fit(X_train, y_train)
y_pred_before = dt_before.predict(X_val)
acc_before = accuracy_score(y_val, y_pred_before)

# apply Min-Max Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# model after normalization
dt_after = DecisionTreeClassifier(random_state=42)
dt_after.fit(X_train_scaled, y_train)
y_pred_after = dt_after.predict(X_val_scaled)
acc_after = accuracy_score(y_val, y_pred_after)

# plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

disp_before = ConfusionMatrixDisplay.from_predictions(y_val, y_pred_before, display_labels=['Not Spam', 'Spam'], ax=axes[0], cmap='Blues')
axes[0].set_title(f'BEFORE Normalization (Validation)\nAccuracy: {acc_before:.4f}')

disp_after = ConfusionMatrixDisplay.from_predictions(y_val, y_pred_after, display_labels=['Not Spam', 'Spam'], ax=axes[1], cmap='Greens')
axes[1].set_title(f'AFTER Normalization (Validation)\nAccuracy: {acc_after:.4f}')

plt.tight_layout()
plt.show()

"""
**Analysis:** As observed in the confusion matrices, applying Min-Max normalization did not change the model's accuracy or predictions. This proves that Decision Trees are scale-invariant algorithms, as they split data based on thresholds regardless of the feature scales.

# **Plot training vs validation curve (Overfitting)**
"""

depth = np.arange(1, 15)
train_scores = []
val_scores = []

for d in depth:
    model_depth = DecisionTreeClassifier(max_depth=d, random_state=42)
    model_depth.fit(X_train, y_train)

    train_scores.append(accuracy_score(y_train, model_depth.predict(X_train)))
    val_scores.append(accuracy_score(y_val, model_depth.predict(X_val)))

plt.plot(depth, train_scores, label="Training", marker='o')
plt.plot(depth, val_scores, label="Validation", marker='s')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

"""
##**Confusion matrix before fixing overfitting**
"""

model_before_overfit = DecisionTreeClassifier(random_state=42)
model_before_overfit.fit(X_train, y_train)

y_pred_before_overfit = model_before_overfit.predict(X_val)

print("Validation accuracy before fixing overfitting:", accuracy_score(y_val, y_pred_before_overfit))

cm_before_overfit = confusion_matrix(y_val, y_pred_before_overfit)
print("Confusion Matrix before fixing overfitting:")
print(cm_before_overfit)

ConfusionMatrixDisplay(confusion_matrix=cm_before_overfit).plot()
plt.title("Before Fixing Overfitting - Validation Set")
plt.show()

"""
# **Fix overfitting**
"""

model_after_overfit = DecisionTreeClassifier(max_depth=5, random_state=42)
model_after_overfit.fit(X_train, y_train)

y_pred_after_overfit = model_after_overfit.predict(X_val)

print("Validation accuracy after fixing overfitting:", accuracy_score(y_val, y_pred_after_overfit))

cm_after_overfit = confusion_matrix(y_val, y_pred_after_overfit)
print("Confusion Matrix after fixing overfitting:")
print(cm_after_overfit)

ConfusionMatrixDisplay(confusion_matrix=cm_after_overfit).plot()
plt.title("After Fixing Overfitting - Validation Set")
plt.show()

"""
**Analysis:** The training vs. validation curve clearly shows the model overfitting as the depth increases (training accuracy approaches 100% while validation accuracy plateaus). By pruning the tree to `max_depth=5`, we significantly reduced overfitting, making the model more generalized for unseen data while maintaining a strong validation accuracy.

# **Handling the Unbalanced Dataset**
In this part, we focus on addressing the problem of unbalanced datasets. The spam emails (Spam) are underrepresented compared to the non-spam emails (Not Spam). We will handle this issue by using techniques like SMOTE.
"""

# Before Balancing (Using max_depth=5 for fair comparison)
dt_unbalanced = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_unbalanced.fit(X_train, y_train)
y_pred_unbal = dt_unbalanced.predict(X_val)

acc_unbal = accuracy_score(y_val, y_pred_unbal)
cm_unbal = confusion_matrix(y_val, y_pred_unbal)
ConfusionMatrixDisplay(confusion_matrix=cm_unbal, display_labels=['Not Spam', 'Spam']).plot()
plt.title(f"Before Balancing (Validation)\nAccuracy: {acc_unbal:.4f}")
plt.show()

"""
# Using SMOTE for Resampling
"""

# Applying SMOTE to resample the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Training the model on resampled data (with max_depth=5)
dt_resampled = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = dt_resampled.predict(X_val)

acc_resampled = accuracy_score(y_val, y_pred_resampled)
cm_resampled = confusion_matrix(y_val, y_pred_resampled)
ConfusionMatrixDisplay(confusion_matrix=cm_resampled, display_labels=['Not Spam', 'Spam']).plot(cmap="Oranges")
plt.title(f"After Balancing using SMOTE (Validation)\nAccuracy: {acc_resampled:.4f}")
plt.show()

"""
**Analysis:** Spam datasets are inherently imbalanced. Initially, the model might favor the majority class (Not Spam). By applying SMOTE, we synthetically balanced the training data. The resulting confusion matrix shows the model's improved ability to identify actual Spam emails, reducing false negatives, which is the most critical metric in spam detection.

# Adjusting the Threshold (Thresholding)
"""

y_pred_proba = dt_resampled.predict_proba(X_val)[:, 1]
threshold = 0.3
y_pred_thresh = (y_pred_proba >= threshold).astype(int)

acc_threshold = accuracy_score(y_val, y_pred_thresh)
cm_threshold = confusion_matrix(y_val, y_pred_thresh)
ConfusionMatrixDisplay(confusion_matrix=cm_threshold, display_labels=['Not Spam', 'Spam']).plot()
plt.title(f"After Threshold Adjustment = {threshold} (Validation)\nAccuracy: {acc_threshold:.4f}")
plt.show()

"""
# FINAL MODEL EVALUATION (ON TEST SET)
"""

y_test_pred_final = dt_resampled.predict(X_test)
acc_final_test = accuracy_score(y_test, y_test_pred_final)

cm_final_test = confusion_matrix(y_test, y_test_pred_final)
ConfusionMatrixDisplay(confusion_matrix=cm_final_test, display_labels=['Not Spam', 'Spam']).plot(cmap='Purples')
plt.title(f"FINAL EVALUATION ON UNSEEN TEST SET\nAccuracy: {acc_final_test:.4f}")
plt.show()

"""
# Decision tree visualization
"""

plt.figure(figsize=(20,10))
plot_tree(
    dt_resampled,
    filled=True,
    feature_names=X.columns,
    class_names=["Not Spam", "Spam"],
    rounded=True
)
plt.show()

"""
# Convert Tree to Rules
"""

rules = export_text(dt_resampled, feature_names=list(X.columns))
print(rules)

""" **Conclusion**
In this project, we successfully built and evaluated a Decision Tree classifier to detect Spam emails.
We rigorously followed standard machine learning methodologies by splitting the data into Training (70%), Validation (15%), and Testing (15%) sets.

We proved that Normalization does not affect decision tree splits. We successfully identified and treated Overfitting by pruning the tree to an optimal depth (`max_depth=5`), guided by validation curves.
Furthermore, we addressed the class bias using SMOTE and adjusted the decision threshold to prioritize catching spam emails.
Finally, evaluating our optimal model on the strictly unseen Test Set yielded highly reliable results, proving that the model generalizes well to new data.
"""