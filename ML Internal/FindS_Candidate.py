import pandas as pd
data=pd.read_csv("/content/EnjoySport.csv")
df=pd.DataFrame(data)
print(df)
def find_s_algorithm(df):
    hypothesis=['0']*(len(df.columns)-1)
    for i in range(len(df)):
        if df.iloc[i,-1]=='Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j]=='0':
                    hypothesis[j]=df.iloc[i,j]
                elif hypothesis[j]!=df.iloc[i,j]:
                    hypothesis[j]='?'
    return hypothesis
hypothesis=find_s_algorithm(df)
print("Final hypothesis by find_s_algorithm: ",hypothesis)






import numpy as np
import pandas as pd
data=pd.read_csv('/content/EnjoySport.csv')
concepts=np.array(data.iloc[:,0:-1])
target=np.array(data.iloc[:,-1])
def learn(concepts,target):
    for i in range(len(target)):
        if target[i]=="Yes":
            specific_h=concepts[i].copy()
            break;
    print("\nInitialization\n")
    print(specific_h)
    general_h=[["?" for i in range(len(specific_h))]for i in range(len(specific_h))]
    print(general_h)
    for i,h in enumerate(concepts):
        if(target[i]=="Yes"):
           for x in range(len(specific_h)):
            if h[x]!=specific_h[x]:
                specific_h[x]='?'
                general_h[x][x]='?'
               
        if target[i]=="No":
            for x in range(len(specific_h)):
                 if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                 else:
                    general_h[x][x]="?"

    indices=[i for i,val in enumerate(general_h) if val==["?"]*len(specific_h)]   
    for i in indices:
        general_h.remove(["?"]*len(specific_h))
    return specific_h,general_h
s_final,g_final=learn(concepts,target)
print("\nFinal specific hypothesis : ",s_final,sep="\n")
print("\nFinal general hypothesis : ",g_final,sep="\n")