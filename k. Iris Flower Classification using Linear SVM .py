from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precisiion",precision_score(y_test,y_pred,average='macro'))
print("F1 score",f1_score(y_test,y_pred,average='macro'))
print("Recall",recall_score(y_test,y_pred,average='macro'))
print("confusion matrix",confusion_matrix(y_test,y_pred))
print("prediction:",iris.target_names[model.predict([[5.1,3.5,1.4,0.2]])[0]])
