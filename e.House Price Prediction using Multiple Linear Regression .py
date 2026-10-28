from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
data=fetch_california_housing()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("mean absolute error",mean_absolute_error(y_test,y_pred))
print("mean square error",mean_squared_error(y_test,y_pred))
print("root mean square error",np.sqrt(mean_squared_error(y_test,y_pred)))
print("r2 error",r2_score(y_test,y_pred))
plt.scatter(y_test,y_pred)
plt.show()
print(model.coef_)
i=np.argmax(np.abs(model.coef_))
print(data.feature_names[i])
