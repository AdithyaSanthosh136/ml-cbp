from sklearn.datasets import load_wine,make_moons
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import numpy as np
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
for i in ['rbf','poly','linear']:
  model=SVC(kernel=i)
  model.fit(x_train,y_train)
  print("Accuracy for "+i+":",accuracy_score(y_test,model.predict(x_test)))

x,y=make_moons(n_samples=300,noise=0.2,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
xx,yy=np.meshgrid(
    np.linspace(x[:,0].min()-0.5,x[:,0].max()+0.5,200),
    np.linspace(x[:,1].min()-0.5,x[:,1].max()+0.5,200),
)
for i in ['rbf','poly','linear']:
  model=SVC(kernel=i)
  model.fit(x_train,y_train)
  print("Accuracy for "+i+":",accuracy_score(y_test,model.predict(x_test)))
  z=model.predict(np.c_[xx.ravel(),yy.ravel()])
  z=z.reshape(xx.shape)
  plt.contourf(xx,yy,z,alpha=0.3)
  plt.scatter(x[:,0],x[:,1],c=y)
  plt.title(i+"kernel")
  plt.show()
