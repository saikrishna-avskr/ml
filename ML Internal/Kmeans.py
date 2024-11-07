from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\cvr\\Desktop\\Mall_Customers.csv")
X = data.drop('CustomerID', axis=1)
X = pd.get_dummies(X, columns=['Gender'])

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
plt.figure(figsize=(8, 6))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'],c=kmeans.labels_, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.scatter(kmeans.cluster_centers_[:, X.columns.get_loc('Annual Income (k$)')], 
            kmeans.cluster_centers_[:, X.columns.get_loc('Spending Score (1-100)')], 
            s=200, c='black', marker='*', label='Centroids')
plt.show()