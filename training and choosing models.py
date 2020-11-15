#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#reading the dataframe
df=pd.read_csv('train.csv')
print(df.shape)
print(df.head)


# In[2]:


#converting category to numerical values
cat={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
df["category"]=df["category"].map(cat)
print(df)


# In[3]:


#removing rows with 'F' in place of numerical values
df=df[df.views!='F']
df=df[df.likes!='F']
df=df[df.dislikes!='F']
df=df[df.comment!='F']
df.shape


# In[4]:


#to change string to numeric values
arr=['adview','views','likes','dislikes','comment']
for i in arr:
    df[i]=df[i].astype(int)


# In[5]:


#encoding published, vidid
from sklearn.preprocessing import LabelEncoder
arr=['vidid','published']
for i in arr:
    df[i]=LabelEncoder().fit_transform(df[i])
print(df)


# In[6]:


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
print(df)


# In[7]:


import matplotlib.pyplot as plt
plt.plot(df['adview'])
plt.show()


# In[8]:


#1 adview entry to high: anomoly so remove
df=df[df['adview']<2000000]
print(df.shape)


# In[9]:


plt.plot(df['views'])
plt.show()


# In[10]:


plt.plot(df['likes'])
plt.show()


# In[11]:


plt.plot(df['dislikes'])
plt.show()


# In[12]:


#to split data into training and testing
Y_train=pd.DataFrame(data=df.iloc[:,1].values,columns=['target'])
df=df.drop(['vidid'],axis=1)
df=df.drop(['adview'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=(train_test_split(df, Y_train,test_size=0.2,random_state=42))
print(x_train.shape)
print(Y_train)


# In[13]:


print(x_train)
#to convert resultant pandas array to numpy array and normalise all attributes
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
print(x_train)


# In[14]:


#a common function to calculate all types of error metrics
import numpy as np
from sklearn import metrics
def error_calc(model):
    p=model.predict(x_test)
    print('Mean Absolute Error',metrics.mean_absolute_error(y_test,p))
    print('Mean Squared Error',metrics.mean_squared_error(y_test,p))
    print('Root Mean Squared Error',np.sqrt(metrics.mean_squared_error(y_test,p)))


# In[15]:


#linear regression and its errors
from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)
error_calc(lr)


# In[16]:


#decsion tree with its errors
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
error_calc(dt)


# In[17]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=250,max_depth=45,min_samples_split=10,min_samples_leaf=2)
rf.fit(x_train,y_train)
error_calc(rf)


# In[18]:


#svr with its errors
from sklearn.svm import SVR
svr=SVR()
svr.fit(x_train,y_train)
error_calc(svr)

#svr best so choosing, and saving it
import joblib
joblib.dump(svr, 'Final_SVR.sav')


# In[19]:


#ann
import keras
from keras.layers import Dense
ann=keras.models.Sequential([
    Dense(8,activation='relu',input_shape=x_train.shape[1:]),Dense(7,activation='relu'),Dense(6,activation='relu'),Dense(1)
])
optimizer=keras.optimizers.Adam()
loss=keras.losses.mean_squared_error
ann.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error'])
history=ann.fit(x_train,y_train,epochs=100)


# In[20]:


ann.summary()


# In[21]:


error_calc(ann)


# In[ ]:





# In[ ]:




