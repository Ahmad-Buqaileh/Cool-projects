import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('../DATA/Mall_Customers.csv')

print(df.head())

print(df.isnull().sum())

print(df.info())

X = df.iloc[:, [3, 4]].values

sc = StandardScaler()
X = sc.fit_transform(X)

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
prediction = kmeans.fit_predict(X)

print(prediction)

plt.scatter(X[prediction == 0, 0], X[prediction == 0, 1], s=100, c='red', label="cluster 1")
plt.scatter(X[prediction == 1, 0], X[prediction == 1, 1], s=100, c='blue', label="cluster 2")
plt.scatter(X[prediction == 2, 0], X[prediction == 2, 1], s=100, c='green', label="cluster 3")
plt.scatter(X[prediction == 3, 0], X[prediction == 3, 1], s=100, c='cyan', label="cluster 4")
plt.scatter(X[prediction == 4, 0], X[prediction == 4, 1], s=100, c='magenta', label="cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="centroids")
plt.title("cluster of customers")
plt.xlabel("Annual Income {k$}")
plt.ylabel("spending score {1-100}")
plt.show()

score = silhouette_score(X, prediction)
print("Silhouette Score: ", score)
