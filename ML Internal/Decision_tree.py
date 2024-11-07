import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\cvr\\Documents\\PlayTennis.csv")
X = data.drop('play', axis=1)
X = pd.get_dummies(X)
y = data['play']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
gini_tree = DecisionTreeClassifier(criterion='gini')
gini_tree.fit(X_train, y_train)
gini_y_pred = gini_tree.predict(X_test)
gini_accuracy = accuracy_score(y_test, gini_y_pred)
print("Accuracy using Gini index:", gini_accuracy)
entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train, y_train)
entropy_y_pred = entropy_tree.predict(X_test)
entropy_accuracy = accuracy_score(y_test, entropy_y_pred)
print("Accuracy using Entropy:", entropy_accuracy)
plt.figure(figsize=(20, 8))
tree.plot_tree(gini_tree, filled=True, feature_names=X.columns, class_names=y.unique().astype(str))
plt.title("Decision Tree using Gini Index")
plt.show()
plt.figure(figsize=(20, 8))
tree.plot_tree(entropy_tree, filled=True, feature_names=X.columns, class_names=y.unique().astype(str))
plt.title("Decision Tree using Entropy")
plt.show()
