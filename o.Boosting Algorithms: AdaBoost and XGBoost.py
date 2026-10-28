from sklearn.datasets  import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from  xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
wine=load_wine()
x,y=wine.data,wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=AdaBoostClassifier(n_estimators=50,random_state=20)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy for adaboost:",accuracy_score(y_test,y_pred))

model=XGBClassifier(n_estimators=50,eval_metric='mlogloss',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy for xgboost:",accuracy_score(y_test,y_pred))

model=RandomForestClassifier(n_estimators=50,random_state=20)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy for random forest:",accuracy_score(y_test,y_pred))
