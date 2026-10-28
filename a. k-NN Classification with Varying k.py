from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
k=[1,3,5,7,9]
acc=[]
for i in k:
  model=KNeighborsClassifier(n_neighbors=i)
  model.fit(x_train,y_train)
  y_pred=model.predict(x_test)
  a=accuracy_score(y_test,y_pred)
  acc.append(a)
  print("Accuracy for k=",i,"is",a)
best=k[acc.index(max(acc))]
print("best k",best)
plt.plot(k,acc)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()
