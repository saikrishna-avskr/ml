import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('/content/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  
data.columns = ['label', 'text'] 
data['label'] = data['label'].map({'ham': 0, 'spam': 1}) 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
spam_emails = data[data['label'] == 1] 
spam_emails.to_csv('spam_emails.csv', index=False)  
print("Spam emails have been saved to 'spam_emails.csv'.")