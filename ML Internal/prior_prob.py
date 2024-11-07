import pandas as pd
import numpy as np
data=pd.read_csv('/content/creditcard.csv')
target = data['Class']
unique, counts = np.unique(target, return_counts=True)
print(counts)
priors = counts/len(target)
for l,p in zip(unique, priors):
  print(f'{l}: {p}')

output:-
[284315    492]
0: 0.9982725143693799
1: 0.001727485630620034