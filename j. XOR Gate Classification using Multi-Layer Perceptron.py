from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import * 
x=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]
model=MLPClassifier( hidden_layer_sizes=(4,), activation='tanh',
solver='lbfgs', max_iter=1000,random_state=42)
model.fit(x,y)
pred=model.predict(x)
print("Actual output:",y)
print("Predicted Output:",pred)
print("Accuracy;",accuracy_score(y,pred))

model=Perceptron()
model.fit(x,y)
pred=model.predict(x)
print("Actual output:",y)
print("Predicted Output:",pred)
print("Accuracy;",accuracy_score(y,pred))
