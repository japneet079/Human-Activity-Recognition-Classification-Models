import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

#Importing file
sheet1 = "train.csv"
data2 = pd.read_csv(sheet1)

X = data2.drop(columns=["Activity", "subject"]) 
y = data2['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtree = DecisionTreeClassifier(random_state=42)
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