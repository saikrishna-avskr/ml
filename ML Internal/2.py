1.a)
import pandas as pd
from sklearn.preprocessing import*
df=pd.read_csv('/content/Book1.csv')
scalar=StandardScaler()
print('Original dataset')
print(df)
num=df.select_dtypes(include=['float64','int64']).columns
df[num]=scalar.fit_transform(df[num])
print('After scaling')
print(df)
b)
import pandas as pd
from sklearn.preprocessing import*
df=pd.read_csv('/content/Book1.csv')
scalar=MinMaxScaler()
print('Original dataset')
print(df)
num=df.select_dtypes(include=['float64','int64']).columns
df[num]=scalar.fit_transform(df[num])
print('After scaling')
print(df)
c)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import*
df=pd.read_csv('/content/Book1.csv')
summary=df.describe()
print(summary)
plt.figure(figsize=(10,6))
df.hist(bins=10,figsize=(10,6))
plt.suptitle("Histogram")
plt.show()
df.plot(kind='box',subplots=True,layout=(1,len(df.columns)),figsize=(10,6))
plt.suptitle("Boxplot")
plt.show()

d)
import pandas as pd
from sklearn.preprocessing import*
df=pd.read_csv('/content/Book1.csv')
print('orginal dataset')
print(df)
de_duplicate=df.drop_duplicates()
print('After removing duplicates')
print(de_duplicate)

e)
import pandas as pd
df=pd.read_csv("/content/Book1.csv")
print('original dataset')
print(df)
df['Telugu'].fillna(df['Telugu'].mean(),inplace=True)
df['Hindi'].fillna(df['Hindi'].median(),inplace=True)
df['Maths'].fillna(df['Maths'].mode()[0],inplace=True)
print('dataset after imputation')
print(df)
df.to_csv('output.csv',index=False)