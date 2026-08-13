import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

#importing file
sheet1 = "train.csv"
data1 = pd.read_csv(sheet1)

#Setting X for reference and Y to be predicted
X = data1.drop(columns=["Activity", "subject"])
y = data1["Activity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtree = RandomForestClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred))

# subset_actual = y_test.values[:40]
# subset_pred = y_pred[:40]

# categories = np.unique(y_test)

# fig, axes = plt.subplots(nrows=len(categories), ncols=1, figsize=(14, 18), sharex=True)
# fig.suptitle('Decision Tree: Actual vs Predicted Sensor Data (First 40 Rows)', fontsize=16, y=0.98)

# for i, category in enumerate(categories):
#     actual_binary = (subset_actual == category).astype(int)
#     pred_binary = (subset_pred == category).astype(int)
    
#     axes[i].plot(actual_binary, label='Actual', color='crimson', marker='o', linestyle='-', linewidth=2)
#     axes[i].plot(pred_binary, label='Predicted', color='royalblue', marker='X', linestyle='--', linewidth=2)
    
#     axes[i].set_title(f'Activity: {category}', fontweight='bold')
#     axes[i].set_yticks([0, 1])
#     axes[i].set_yticklabels(['No', 'Yes'])
#     axes[i].legend(loc='center right')
#     axes[i].grid(True, linestyle=':', alpha=0.7)

# plt.xlabel('Test Set Item Number')
# plt.tight_layout()
# plt.show()

# #Splitting the data, 80 percent for training and 20 percent for testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# #Initialize and train Random Forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# #Predict Y
# y_pred = rf.predict(X_test)

# #Returning Solution
# print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
# print("Detailed Report:\n", classification_report(y_test, y_pred))

#Comparing Actual vs Predicted
# subset_actual = y_test.values[:40]
# subset_pred = y_pred[:40]
# categories = rf.classes_

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
# fig.suptitle('Random Forest: Actual vs Predicted (First 40 Orders)', fontsize=14, y=0.98)

# for i, category in enumerate(categories):
#     actual_binary = (subset_actual == category).astype(int)
#     pred_binary = (subset_pred == category).astype(int)
    
#     # Plot Actual (Crimson) and Predicted (Blue)
#     axes[i].plot(actual_binary, label='Actual', color='crimson', marker='o', linestyle='-', linewidth=2)
#     axes[i].plot(pred_binary, label='Predicted', color='royalblue', marker='X', linestyle='--', linewidth=2)
    
#     axes[i].set_title(f'Category: {category}', fontweight='bold')
#     axes[i].set_yticks([0, 1])
#     axes[i].set_yticklabels(['No', 'Yes'])
#     axes[i].legend(loc='center right')
#     axes[i].grid(True, linestyle=':', alpha=0.7)

# plt.xlabel('Test Set Item Number')
# plt.tight_layout()
# plt.show()