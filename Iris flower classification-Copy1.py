#!/usr/bin/env python
# coding: utf-8

# Iris Flower Classifcation

# 1. Importing modules & analyzing dataset

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# #### 2. Importing dataset
Ir=pd.read_csv('Iris.csv')
Ir.head()


# ## Analyzing Model



Ir=Ir.drop(columns = ['Id'])
Ir.head()


# ### Statistics of dataset

Ir.describe()




# no of samples in each class
Ir['Species'].value_counts()





# check null values
Ir.isnull().sum()


#  Graphs 

# 1. Histograms 




# sepal length
Ir['SepalLengthCm'].hist()





# sepal width
Ir['SepalWidthCm'].hist()





#petal length
Ir['PetalLengthCm'].hist()





# petal width
Ir['PetalWidthCm'].hist()


# 2. Scatter Plots




# Categorising data for scatter plot
colors=['yellow','green','purple']
Species= ['Iris-virginica','Iris-versicolor' , 'Iris-setosa']


 


# sepal length vs sepal width
for i in range(3):
    x=Ir[Ir['Species']==Species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c=colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()    





# petal length vs petal width
for i in range(3):
    x=Ir[Ir['Species']==Species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c=colors[i], label=Species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()    





# sepal length vs petal length
for i in range(3):
    x=Ir[Ir['Species']==Species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'], c=colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()    





#correlation 
Ir.corr()





corr=Ir.corr()
fig, ax= plt.subplots(figsize=(3,3))
sns.heatmap(corr, annot=True, ax=ax)





#label encoder= convert data into machine understandable form
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()





Ir['Species']= le.fit_transform(Ir['Species'])
Ir.head()


# Model Training


from sklearn.model_selection import train_test_split
#train 60
#test 40
X = Ir.drop(columns=['Species'])
Y = Ir['Species']
x_train, x_test, y_train,y_test= train_test_split(X,Y, test_size=0.40)


# Logistic Regression 



from sklearn.linear_model import LogisticRegression
model= LogisticRegression()





model.fit(x_train, y_train)




#print metric to get performance
print("Accuracy",model.score(x_test,y_test)*100)



# knn= k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

model.fit(x_train, y_train)


print("Accuracy", model.score(x_test,y_test)*100)

#decision tree

from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()


model.fit(x_train,y_train)

print("Accuracy", model.score(x_test,y_test)*100)
