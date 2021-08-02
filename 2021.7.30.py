# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:06:13 2021

@author: dell
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
path="C:/Users/dell/Desktop/iris.csv"
data=pd.read_csv(path,index_col=0)#index_col=0什么意思来着
data.columns
data_v=data.values
data.iloc[0,0]
data.iloc[0,:]
data.iloc[:,-1]

model=KNeighborsClassifier(n_neighbors=5,p=1)
x=data.iloc[:,0:4]
y=data.iloc[:,4]


model.fit(x,y)
model.score(x,y)

area1=data.iloc[:,0]-data.iloc[:,1]
area2=data.iloc[:,2]-data.iloc[:,3]
data["area1"]=area1
data["area2"]=area2
data_final=data.iloc[:,[0,1,2,3,5,6,4]]
save_path="C:/Users/dell/Desktop/iris_new.csv"
data_final.to_csv(save_path)
x=data_final.iloc[:,:-1]
y=data_final.iloc[:,-1]
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.4)
model=KNeighborsClassifier(n_neighbors=1)
model.fit(train_x,train_y)
print(model.score(train_x,train_y))
print(model.score(test_x,test_y))

#梯度下滑
def f(x):
    return x*x-2*x-5
    
def fedriv(x):
    return 2*x-2
learning_rate=0.05
n_iter=100

xs=np.zeros(n_iter+1)
xs[0]=100
for i in range(n_iter):
    xs[i+1]=xs[i]-learning_rate*fedriv(xs[i])
plt.plot(xs)

from scipy.optimize import minimize
a=minimize(f,x0=100).x


def f2(x):
    return np.exp(-x**2)*(x**2)
from scipy.optimize import minimize
a1=minimize(f2,x0=3).x


def E(x):
    a=-x*np.log(x)-(1-x)*np.log(1-x)
    return a
x=np.linspace(0.01,0.99,100)
y=E(x)
plt.plot(x,y)

def G(x):
    a=1-x*x-(1-x)*(1-x)
    return a
x=np.linspace(0.01,0.99,100)
y=G(x)
plt.plot(x,y)



#

path=r"C:\Users\dell\Desktop\wm.csv"
wm=pd.read_csv(path,index_col=0)
x=wm.iloc[:,:6]
y=wm.iloc[:,-1]
import sklearn.tree as tree
model=tree.DecisionTreeClassifier(criterion="entropy",max_depth=2)
model.fit(x,y)
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
bc=load_breast_cancer()
x=bc.data
y=bc.target
model.fit(x_train,y_train)
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)
model.score(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))


model_1=KNeighborsClassifier(n_neighbors=1)
model_1.fit(x_train,y_train)
print(model_1.score(x_test, y_test))
print(model_1.score(x_train, y_train))


from sklearn.linear_model import LogisticRegression
model_2=LogisticRegression()
model_2.fit(x_train,y_train)
print(model_2.score(x_test, y_test))
print(model_2.score(x_train, y_train))




sample_size=100000
tmp=np.random.normal(-10,10,(sample_size,2))#normal&uniform
plt.scatter(tmp[:,0],tmp[:,1])
radius=7
label=np.zeros(sample_size)
for i in range(sample_size):
    if np.sqrt(tmp[i,0]**2+tmp[i,1]**2)<radius:
        label[i]=1
    else:
        label[i]=0
  
plt.figure(figsize=(10,10))
plt.scatter(tmp[label==0,0],tmp[label==0,1],color="blue")
plt.scatter(tmp[label==1,0],tmp[label==1,1],color="pink")
