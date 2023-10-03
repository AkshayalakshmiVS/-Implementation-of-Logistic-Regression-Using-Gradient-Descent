# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Akshaya Lakshmi VS 
RegisterNumber:  212222040005
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
Array values of x

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/de5e8a03-9bae-4d4a-979e-088ead3cad76)

Array values of y

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/c50226ec-72fc-43d3-b951-2a3195ad42c2)

Exam 1 - score graph

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/ec307286-f430-4c15-baa2-8d3edf368302)

Sigmoid function graph

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/e6ae5661-edd3-45b0-ba6e-f8124ee21f0c)

x_train_grad value

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/2f87ed74-f0c7-4aa2-855f-ee7de9af6b72)

y_train_grad value

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/5b20fab0-8969-4e76-aed1-f381c5c43082)

Print res.x

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/295bd33b-d2b4-491d-85ae-84e4d8fa01c2)

Decision boundary - graph for exam score

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/34d98d0d-8f44-44d2-9607-c145759bf8cc)

Probability value

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/da042e43-0af6-4d57-bbb9-308add9ed88d)

Prediction value of mean

![image](https://github.com/Deeksha78/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128116204/519c5a2b-ea3b-44f7-9947-7c2f624ca059)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

