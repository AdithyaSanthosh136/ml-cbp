from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import *
iris=load_iris()
x=iris.data[:,:2]
model1=KMeans(n_clusters=3,random_state=42,n_init=10)
p1=model1.fit_predict(x)
print("silhouette score for kmeans",silhouette_score(x,p1))
model2=GaussianMixture(n_components=3,random_state=42)
model2.fit(x)
p2=model2.predict(x)
print("silhouette score for gaussian mixture",silhouette_score(x,p2))
plt.scatter(x[:,0],x[:,1],c=p1)
plt.scatter(model1.cluster_centers_[:,0],model1.cluster_centers_[:,1],marker='X',s=150)
plt.show()
plt.scatter(x[:,0],x[:,1],c=p2)
plt.scatter(model2.means_[:,0],model2.means_[:,1],marker='X',s=150)
plt.show()
