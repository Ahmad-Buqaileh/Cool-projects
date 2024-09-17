# Project Overview
### The objective of this project is to cluster customers based on their annual income and spending score using the K-Means clustering algorithm.
### The dataset consists of mall customers and their various attributes. We aim to understand customer segments, which can help in customer behavior analysis, targeted marketing, and improving customer experiences.
### The final outcome of this project is a set of clusters that groups customers based on similar income and spending patterns, along with an assessment of the clustering performance using the silhouette score.
# First, try doing it on your own. If you struggle with something, you can find the steps outlined below.
## Import necessary libraries
```bash
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```
## Load the dataset
```bash
df = pd.read_csv('../DATA/Mall_Customers.csv')
```
#### Display the first few rows of the dataset to understand its structure
```bash
print(df.head())
```
output :
```bash
   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40
```
#### Check for missing values to ensure data quality
```bash
print(df.isnull().sum())
```
output :
```bash
CustomerID                0
Gender                    0
Age                       0
Annual Income (k$)        0
Spending Score (1-100)    0
dtype: int64
```
#### Display dataset information to understand its data types and structure
```bash
print(df.info())
```
output :
```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Gender                  200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 7.9+ KB
None
```
## Extract the relevant features for clustering: Annual Income (column 3) and Spending Score (column 4)
```bash
X = df.iloc[:, [3, 4]].values
```
## Scale the data to ensure that both features contribute equally to the distance metric used by K-Means
```bash
sc = StandardScaler()
X = sc.fit_transform(X)
```
## Use the Elbow method to find the optimal number of clusters (between 1 and 14 clusters)
### # List to store Within-Cluster Sum of Squares (WCSS) for each number of clusters
```bash
wcss = []
for i in range(1, 15):
    # KMeans initialization
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    # Fit KMeans with i clusters
    kmeans.fit(X)
    # Append the WCSS for each cluster number
    wcss.append(kmeans.inertia_) 
```
## Plot the Elbow method result to visualize the optimal number of clusters
```bash
plt.plot(range(1, 15), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS") 
plt.show()
```
output :

![image](https://github.com/user-attachments/assets/91bd54f9-c9fb-46a3-bd18-0e5a81ff1d88)
## Based on the Elbow method, we select 5 clusters and apply K-Means
```bash
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
```
#### Fit and predict the cluster for each data point
```bash
prediction = kmeans.fit_predict(X)
```
## Print the cluster assignments for each data point
```bash
print(prediction)
```
output :
```bash
[3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3
 4 3 4 3 4 3 0 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 0 1 2 1 2 1 0 1 2 1 2 1 2 1 2 1 0 1 2 1 2 1
 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1]
```
## Visualize the clusters by plotting them on a 2D plot
#### Each cluster is colored differently
```bash
plt.scatter(X[prediction == 0, 0], X[prediction == 0, 1], s=100, c='red', label="Cluster 1")
plt.scatter(X[prediction == 1, 0], X[prediction == 1, 1], s=100, c='blue', label="Cluster 2")
plt.scatter(X[prediction == 2, 0], X[prediction == 2, 1], s=100, c='green', label="Cluster 3")
plt.scatter(X[prediction == 3, 0], X[prediction == 3, 1], s=100, c='cyan', label="Cluster 4")
plt.scatter(X[prediction == 4, 0], X[prediction == 4, 1], s=100, c='magenta', label="Cluster 5")
```
#### Plot the cluster centroids as yellow points
```bash
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="Centroids")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income {k$}")
plt.ylabel("Spending Score {1-100}")
plt.legend()
plt.show()
```
output:

![image](https://github.com/user-attachments/assets/fb2f8c80-1e22-4a10-861a-3298ecbdbc4a)
## Calculate the silhouette score to evaluate how well the clusters are formed
```bash
score = silhouette_score(X, prediction)
print("Silhouette Score: ", score)
```
output :
```bash
Silhouette Score:  0.5546571631111091
```
# Conclusion
### he project uses K-Means clustering to segment mall customers into distinct groups based on income and spending habits. These customer segments can be used to identify high-spending customers, budget-conscious customers, and others, enabling more targeted marketing strategies. The silhouette score provides a quantitative measure of the clustering quality.
### **Silhouette Score :**
### 0.71 to 1.0: Excellent clustering
### 0.51 to 0.70: Good clustering   <--- Our score is good.
### 0.26 to 0.50: Average clustering
### Below 0.25: Poor clustering












