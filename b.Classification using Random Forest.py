from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precisiion",precision_score(y_test,y_pred,average='macro'))
print("F1 score",f1_score(y_test,y_pred,average='macro'))
print("Recall",recall_score(y_test,y_pred,average='macro'))
print("confusion matrix",confusion_matrix(y_test,y_pred))
plt.bar(iris.feature_names,model.feature_importances_)
plt.show()
