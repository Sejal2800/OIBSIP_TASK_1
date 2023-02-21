#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classifcation

# #### 1. Importing modules & analyzing dataset

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# #### 2. Importing dataset

# In[ ]:


Ir=pd.read_csv('Iris.csv')
Ir.head()


# ## Analyzing Model

# In[ ]:


Ir=Ir.drop(columns = ['Id'])
Ir.head()


# ### Statistics of dataset

# In[ ]:


Ir.describe()


# In[ ]:


# no of samples in each class
Ir['Species'].value_counts()


# In[ ]:


# check null values
Ir.isnull().sum()


# ### Graphs 

# #### 1. Histograms 

# In[ ]:


# sepal length
Ir['SepalLengthCm'].hist()


# In[ ]:


# sepal width
Ir['SepalWidthCm'].hist()


# In[ ]:


#petal length
Ir['PetalLengthCm'].hist()


# In[ ]:


# petal width
Ir['PetalWidthCm'].hist()


# #### 2. Scatter Plots

# In[ ]:


# Categorising data for scatter plot
colors=['yellow','green','purple']
Species= ['Iris-virginica','Iris-versicolor' , 'Iris-setosa']


# In[ ]:


# sepal length vs sepal width
for i in range(3):
    x=Ir[Ir['Species']==Species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c=colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()    


# In[ ]:


# petal length vs petal width
for i in range(3):
    x=Ir[Ir['Species']==Species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c=colors[i], label=Species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()    


# In[ ]:


# sepal length vs petal length
for i in range(3):
    x=Ir[Ir['Species']==Species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'], c=colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()    


# In[ ]:


#correlation 
Ir.corr()


# In[ ]:


corr=Ir.corr()
fig, ax= plt.subplots(figsize=(3,3))
sns.heatmap(corr, annot=True, ax=ax)


# In[ ]:


#label encoder= convert data into machine understandable form
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[ ]:


Ir['Species']= le.fit_transform(Ir['Species'])
Ir.head()


# # Model Training

# In[ ]:


from sklearn.model_selection import train_test_split
#train 60
#test 40
X = Ir.drop(columns=['Species'])
Y = Ir['Species']
x_train, x_test, y_train,y_test= train_test_split(X,Y, test_size=0.40)


# ### Logistic Regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


#print metric to get performance
print("Accuracy",model.score(x_test,y_test)*100)


# In[ ]:


# knn= k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


print("Accuracy", model.score(x_test,y_test)*100)


# In[ ]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


print("Accuracy", model.score(x_test,y_test)*100)


# In[ ]:





# In[ ]:




