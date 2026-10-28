from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
iris=load_iris()
x=iris.data[:,:2]
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=Perceptron()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precisiion",precision_score(y_test,y_pred, average='weighted'))
print("F1 score",f1_score(y_test,y_pred, average='weighted'))
print("Recall",recall_score(y_test,y_pred, average='weighted'))
print("confusion matrix",confusion_matrix(y_test,y_pred))
xx,yy=np.meshgrid(
    np.linspace(x[:,0].min()-0.5,x[:,0].max()+0.5,200),
    np.linspace(x[:,1].min()-0.5,x[:,1].max()+0.5,200)
)
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=0.3)
plt.scatter(x[:,0],x[:,1],c=y)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()
