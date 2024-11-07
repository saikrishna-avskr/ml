import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
import seaborn as sns 
import matplotlib.pyplot as plt 
df = pd.read_csv("C:\\Users\\22B81A6637\\german_credit_data.csv") 
print(df.head()) 
print(df.info()) 
label_encoder = LabelEncoder() 
for column in df.columns: 
    if df[column].dtype == 'object': 
        df[column] = label_encoder.fit_transform(df[column]) 
X = df.drop('Risk', axis=1) 
y = df['Risk'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
# 1. Logistic Regression 
#model = LogisticRegression() 
# 2. Support Vector Machine 
model = SVC() 
# 3. Naive Bayes 
#model = GaussianNB() 
# Train the model 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
cm = confusion_matrix(y_test, y_pred) 
accuracy = accuracy_score(y_test, y_pred) 
report = classification_report(y_test, y_pred) 
print("Confusion Matrix:\n", cm) 
print("\nAccuracy:", accuracy) 
print("\nClassification Report:\n", report) 
plt.figure(figsize=(6, 4)) 
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good']) 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.title("Confusion Matrix") 
plt.show() 
