# Project Overview
### In this project, we will develop and evaluate multiple classification models to predict loan defaults.
### Load the dataset and inspect its structure, data types, and check for missing values.
### Prepare the features and target variable. Split the dataset into training and testing sets for model evaluation.
### Standardize the features to ensure consistent scaling, which helps in improving model performance.
### Train various classification models on the training data to identify the most effective model for predicting loan defaults.
### Assess the performance of each model using metrics such as the confusion matrix, accuracy score, and ROC curve to determine their effectiveness.
### Visualize the results with confusion matrices and ROC curves to compare model performances and understand their strengths and weaknesses.
## **( In the code, I only used Logistic Regression. However, you can see me using multible classifers down bellow )**
# First, try doing it on your own. If you struggle with something, you can find the steps outlined below.

## Import necessary Libraries
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
```
## Load the dataset
#### Replace '../DATA/Default_Fin.csv' with the path to your dataset file
```bash
df = pd.read_csv('../DATA/Default_Fin.csv')
```
#### Display the first few rows of the dataset to understand its structure
```bash
print(df.head())
```
output:
```bash
   Index  Employed  Bank Balance  Annual Salary  Defaulted?
0      1         1       8754.36      532339.56           0
1      2         0       9806.16      145273.56           0
2      3         1      12882.60      381205.68           0
3      4         1       6351.00      428453.88           0
4      5         1       9427.92      461562.00           0
```
#### Display the structure and data types of the dataset
```bash
print(df.info())
```
output:
```bash
 <class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Index          10000 non-null  int64  
 1   Employed       10000 non-null  int64  
 2   Bank Balance   10000 non-null  float64
 3   Annual Salary  10000 non-null  float64
 4   Defaulted?     10000 non-null  int64  
dtypes: float64(2), int64(3)
memory usage: 390.8 KB
None
```
#### Check for missing values in the dataset
```bash
print(df.isnull().sum())
```
output:
```bash
Index            0
Employed         0
Bank Balance     0
Annual Salary    0
Defaulted?       0
dtype: int64
```
## Prepare features (X) and target variable (y)
#### Assuming that the target variable is the last column and features are all other columns except the first one
```bash
X = df.iloc[:, 1:-1].values  # Features
y = df.iloc[:, -1].values    # Target variable
```
## Split the dataset into training and testing sets
#### 80% of the data will be used for training and 20% for testing
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
## Standardize features by removing the mean and scaling to unit variance
#### Fit on training data and transform both training and testing data
```bash
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## Initialize and train the Logistic Regression model
```bash
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```
## Make predictions on the test set
```bash
predictions = classifier.predict(X_test)
```
## Display the predictions alongside the actual values
#### This helps in understanding how many predictions were correct or incorrect
```bash
print(np.concatenate((predictions.reshape(len(predictions), 1), y_test.reshape(len(y_test), 1)), 1))
```
output: 
```bash
[[0 0]
 [0 0]
 [0 0]
 ...
 [0 0]
 [0 0]
 [0 0]]
```
## Evaluate the model using a confusion matrix and accuracy score
```bash
cm = confusion_matrix(y_test, predictions)  # Confusion Matrix
ac = accuracy_score(y_test, predictions)     # Accuracy Score
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1920    6]
 [  51   23]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.9715
```
## Plot the confusion matrix using a heatmap for better visualization
```bash
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```
output:
![image](https://github.com/user-attachments/assets/e407b6af-2ac8-4523-8cb8-2fa12ae2ad91)
## Predict probabilities for the positive class
```bash
y_prob = classifier.predict_proba(X_test)[:, 1]
```
## Compute the ROC curve
```bash
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
```
## Compute the Area Under the ROC Curve (AUC)
```bash
roc_auc = auc(fpr, tpr)
```
### Plot the ROC curve
```bash
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
```
output:
![image](https://github.com/user-attachments/assets/0efcc935-49d1-4c8a-83b0-3f6f70adfe58)
# Now let's see how will the other Classifiers do
## **Decision Tree Classification**
#### Initialize and train the Decision Tree 
```bash
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1886   40]
 [  49   25]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.9555
```
## Plot the confusion matrix
output:
![image](https://github.com/user-attachments/assets/48febbe6-4d33-4025-840d-c23ef2746482)
### Plot the ROC curve
output:
![image](https://github.com/user-attachments/assets/e83ef1dc-48dc-4125-8b5a-e406edf107fb)
## **K-NN**
#### Initialize and train the K-NN
```bash
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1915   11]
 [  50   24]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.9695
```
## Plot the confusion matrix
output:
![image](https://github.com/user-attachments/assets/512625b3-8571-4043-af28-8426fc0c3d30)
### Plot the ROC curve
output:
![image](https://github.com/user-attachments/assets/34cf9e54-86f8-42b7-af71-fbb89c2fd1c4)
## **SVM**
#### Initialize and train the SVM
```bash
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1926    0]
 [  74    0]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.963
```
## Plot the confusion matrix
output:
![image](https://github.com/user-attachments/assets/962340fa-41c2-4f95-b1e0-61a612245956)
## **Random forest Classification**
#### Initialize and train the Random forest Classification
```bash
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1911   15]
 [  47   27]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.969
```
## Plot the confusion matrix
output:
![image](https://github.com/user-attachments/assets/ac8e7b86-3327-4395-9007-0747a942ad6b)
### Plot the ROC curve
![image](https://github.com/user-attachments/assets/f14f1f22-0a67-4aac-891c-07de1a69cf65)
## **Naive bayes**
#### Initialize and train the Naive bayes
```bash
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1911   15]
 [  56   18]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.9645
```
## Plot the confusion matrix
output:
![image](https://github.com/user-attachments/assets/afb7fb48-ff50-488f-8ed9-af63b0d3ff37)
## **Kernel SVM**
#### Initialize and train the Kernel SVM
```bash
from sklearn.svm import SVC

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
#### output the Confusion Matrix
```bash
print(cm)
```
output:
```bash
[[1924    2]
 [  57   17]]
```
#### output the Accuarcy Score
```bash
print(ac)
```
output:
```bash
0.9705
```
## Plot the confusion matrix
output:
![image](https://github.com/user-attachments/assets/86ba3314-c9dc-4120-b6a6-00cf9f4753b7)
# Conclusion
### In this project, we developed and evaluated various classification models to predict loan defaults. We prepared the data, trained multiple classifiers including Logistic Regression, Decision Tree, K-Nearest Neighbors (K-NN), Support Vector Machine (SVM), Random Forest, Naive Bayes, and Kernel SVM, and assessed their performance through confusion matrices, accuracy scores, and ROC curves. This comprehensive evaluation provides insights into the effectiveness of each model in predicting loan defaults.




