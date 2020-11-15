#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
df=pd.read_csv("test.csv")
df


# In[38]:


cat={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
df["category"]=df["category"].map(cat)
print(df)


# In[39]:


df=df[df.views!='F']
df=df[df.likes!='F']
df=df[df.dislikes!='F']
df=df[df.comment!='F']
print(df)


# In[40]:


#converting duration in proper format of seconds
l=[]
q=[]
for i in df['duration']:
    l.append(i)
for i in l:
    j=i[2:len(i)-1]
    if 'M' in i and 'H' in i and 'S' in i:
        j=j.split('H')
        j[1]=j[1].split('M')
        j=int(j[0])*3600+int(j[1][0])*60+int(j[1][1])
    elif 'M' in i and 'H' in i:
        j=j.split('H')
        j=int(j[0])*3600+int(j[1])*60
    elif 'H' in i and 'S' in i:
        j=j.split('H')
        j=int(j[0])*3600+int(j[1])
    elif 'M' in i and 'S' in i:
        j=j.split('M')
        j=int(j[0])*60+int(j[1])
    elif 'H' in i:
        j=int(j)*3600
    elif 'M' in i:
        j=int(j)*60
    elif 'S' in i:
        j=int(j)
    q.append(j)
df['duration']=q

from sklearn.preprocessing import LabelEncoder
arr=['vidid','published']
for i in arr:
    df[i]=LabelEncoder().fit_transform(df[i])

arr=['views','likes','dislikes','comment']
for i in arr:
    df[i]=df[i].astype(int)
    
print(df)


# In[41]:


df.drop(['vidid'],axis=1)
import matplotlib.pyplot as plt
plt.plot(df['views'])
plt.show()
plt.plot(df['likes'])
plt.show()
plt.plot(df['dislikes'])
plt.show()
plt.plot(df['comment'])
plt.show()
plt.plot(df['duration'])
plt.show()


# In[42]:


df=df[df['views']<200000000]
print(df.shape)
plt.plot(df['likes'])
plt.show()
df


# In[43]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=df.iloc[:,1:]
x=scaler.fit_transform(x)
print(x)


# In[44]:


import joblib
loaded_model = joblib.load("Final_SVR.sav")
result = loaded_model.predict(x)
print(result)


# In[45]:


df['adview']=result
print(df)


# In[46]:


df.to_csv('Predictions_Submission.csv') 


# In[ ]:




