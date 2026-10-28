from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as  plt
x,y=make_blobs(n_samples=300,centers=4,random_state=42)
plt.scatter(x[:,0],x[:,1])
plt.show()
wcss=[]
for i in range(1,11):
  model=KMeans(n_clusters=i,n_init=10,random_state=42)
  model.fit(x,y)
  wcss.append(model.inertia_)
plt.plot(range(1,11),wcss)
plt.show()
k=4
model=KMeans(n_clusters=k,n_init=10,random_state=42)
labels=model.fit_predict(x)
print("silhouette score:",silhouette_score(x,labels))
plt.scatter(x[:,0],x[:,1],c=labels)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,marker='X')
plt.show()
