import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


#importing file
sheet1 = "test.csv"
data1 = pd.read_csv(sheet1)

#Setting X for reference and Y to be predicted
X = data1.drop(columns=["Activity", "subject"]) 
y = data1['Activity']

#Splitting the data, 80 percent for training and 20 percent for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

#Scaling the data for more uniform results
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Setting number of neighbours 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

#Predicting the categories
y_pred = knn.predict(X_test_scaled)

#Returning results
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Detailed Report:\n", classification_report(y_test, y_pred))

#Line Chart : Comparing actual to predicted values
# subset_actual = y_test.values[:40]
# subset_pred = y_pred[:40]
# categories = np.unique(y_test)
# fig, axes = plt.subplots(nrows=len(categories), ncols=1, figsize=(14, 18), sharex=True)
# fig.suptitle('Actual vs Predicted Sensor Data (First 40 Rows)', fontsize=16, y=0.98)

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


