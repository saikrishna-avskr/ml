import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv('/content/creditcard.csv')
X = data.drop('Class', axis=1)
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model=RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
td=data.iloc[x_test.index]
fraud = td[y_pred == 1]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
fraud['Amount'] = 0
fraud.to_csv('/content/fraud_transactions.csv', index=False)