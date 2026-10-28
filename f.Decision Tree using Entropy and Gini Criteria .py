from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
wine=load_wine()
x,y=wine.data,wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(criterion='entropy',random_state=30)
print("------------decision tree using entropy------------")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precisiion",precision_score(y_test,y_pred,average='macro'))
print("F1 score",f1_score(y_test,y_pred,average='macro'))
print("Recall",recall_score(y_test,y_pred,average='macro'))
print("------------decision tree using gini------------")
model=DecisionTreeClassifier(criterion='gini',random_state=30)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precisiion",precision_score(y_test,y_pred,average='macro'))
print("F1 score",f1_score(y_test,y_pred,average='macro'))
print("Recall",recall_score(y_test,y_pred,average='macro'))
