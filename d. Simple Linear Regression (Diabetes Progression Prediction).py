from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import numpy as np
data=load_diabetes()
x=data.data[:,2].reshape(-1,1)
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("mean absolute error",mean_absolute_error(y_test,y_pred))
print("mean square error",mean_squared_error(y_test,y_pred))
print("root mean square error",np.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 error",r2_score(y_test,y_pred))
print(model.predict([[0.05]])[0])
plt.scatter(x_test,y_test,label="actual")
plt.scatter(x_test,y_pred,label="predicted")
