from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
iris = load_iris() 
X = iris.data 
y = iris.target 
target_names = iris.target_names 
scaler = StandardScaler() 
X_std = scaler.fit_transform(X) 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_std) 
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2']) 
df_pca['target'] = y 
plt.figure(figsize=(8, 6)) 
for target, color in zip(np.unique(y), ['r', 'g', 'b']): 
    plt.scatter(df_pca[df_pca['target'] == target]['PC1'], 
                df_pca[df_pca['target'] == target]['PC2'], 
                c=color, label=target_names[target]) 
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Component 2') 
plt.legend() 
plt.title('PCA of Iris Dataset') 
plt.show() 
lda = LDA(n_components=2) 
X_lda = lda.fit_transform(X_std, y) 
df_lda = pd.DataFrame(X_lda, columns=['LD1', 'LD2']) 
df_lda['target'] = y 
plt.figure(figsize=(8, 6)) 
for target, color in zip(np.unique(y), ['r', 'g', 'b']): 
    plt.scatter(df_lda[df_lda['target'] == target]['LD1'], 
                df_lda[df_lda['target'] == target]['LD2'], 
                c=color, label=target_names[target]) 
plt.xlabel('Linear Discriminant 1') 
plt.ylabel('Linear Discriminant 2') 
plt.legend() 
plt.title('LDA of Iris Dataset') 
plt.show() 