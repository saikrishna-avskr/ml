import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_methods = ['single', 'complete', 'average']
X=pd.read_csv("/content/Mall_Customers.csv")
X = X.drop('CustomerID', axis=1)
X = pd.get_dummies(X, columns=['Gender'])
print(X.head())
for method in linkage_methods:
    Z = linkage(X, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, p=5, truncate_mode='lastp')
    plt.title(f"Dendrogram using {method} linkage")
    plt.xlabel("Data points")
    plt.ylabel("Distance")
    plt.show()
    

dataset : https://filetransfer.io/data-package/uGMvMEfY#link