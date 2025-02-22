import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Assume 'model' is your trained model
# X_test, y_test are your test data and labels (one-hot encoded)
# Convert y_test to integer labels
y_true = np.argmax(y_test, axis=1)

# Predict probabilities
y_pred_probs = model.predict(X_test)

# Convert predicted probabilities to integer labels
y_pred = np.argmax(y_pred_probs, axis=1)

# 1. Classification Report
# target_names must match the order of your label mapping
target_names = ['Positive', 'Neutral', 'Negative']
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
