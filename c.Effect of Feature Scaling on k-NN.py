from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
wine=load_wine()
x,y=wine.data,wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
model1=KNeighborsClassifier(n_neighbors=5)
model1.fit(x_train,y_train)
y1=model1.predict(x_test)
print("accuracy without scaling:",accuracy_score(y_test,y1))
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
model2=KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train,y_train)
y2=model2.predict(x_test)
print("accuracy with scaling:",accuracy_score(y_test,y2))
