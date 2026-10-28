from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import numpy as np
import seaborn as sns
wine=load_wine()
x,y=wine.data,wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=80)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("classification report:",classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()
errors=np.where(y_pred!=y_test)[0]
if len(errors)==0:
  print("no misclassified classes")
else:
  wrong=y_pred[errors]
  values,counts=np.unique(wrong,return_counts=True)
  print(wine.target_names[values[np.argmax(counts)]])
