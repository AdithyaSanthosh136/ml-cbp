from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
wine=load_wine()
x,y=wine.data,wine.target
model=DecisionTreeClassifier()
score=cross_val_score(model,x,y,cv=5)
for i in range(5):
  print("acuracy of fold"+str(i+1)+":",score[i])
print("avg Accuracy:",np.average(score))
print("standard accuracy",np.std(score))
