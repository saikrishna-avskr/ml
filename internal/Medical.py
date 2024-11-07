dataset:  https://filetransfer.io/data-package/tVtDQkgv#link

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("/content/diabetes_prediction_dataset.csv")
X=data.drop('diabetes',axis=1)
X=pd.get_dummies(X,columns=['gender','smoking_history'])
y=data['diabetes']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
clf=RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
td=data.iloc[X_test.index]
diabetic = td[y_pred == 1]
diabetic.to_csv('diabetic_patients.csv', index=False)
print("Diabetic patients saved to 'diabetic_patients.csv'")