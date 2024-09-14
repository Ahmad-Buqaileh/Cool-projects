# Project Overview
#### The dataset used in this project contains financial indicators of companies, with the target label indicating whether a company went bankrupt or not. The dataset includes various financial ratios and indicators across multiple companies, which serve as the features for predicting bankruptcy.
#### The dataset is loaded and basic exploration is performed to understand the structure, check for missing values, and gain insights.
#### The data is split into training (80%) and test (20%) sets. A Logistic Regression model is trained on the training set. Logistic regression is a suitable choice for binary classification problems like bankruptcy prediction.
#### Predictions are made on the test set, and the accuracy of the model is calculated. Cross-validation with 10 folds is performed to assess the model’s stability and performance across different subsets of the data. The confusion matrix is used to visualize the number of correct and incorrect predictions.
#### A Receiver Operating Characteristic (ROC) curve is plotted, and the Area Under the Curve (AUC) is calculated to measure the model's ability to distinguish between default and non-default companies.
# First, try doing it on your own. If you struggle with something, you can find .....
## **Import Necessary Libraries**
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
```
## **Load the dataset** 
```bash
df = pd.read_csv('../DATA/Company Bankruptcy Prediction.csv')
```
## **Print dataset info**
#### Display the first few rows of the dataset
```bash
print(df.head())
```
output :
```bash
    ROA(C) before interest and depreciation before interest  ...  Bankrupt?
0                                           0.370594         ...          1
1                                           0.464291         ...          1
2                                           0.426071         ...          1
3                                           0.399844         ...          1
4                                           0.465022         ...          1
[5 rows x 96 columns]
```
#### Display a concise summary of the dataset, including data types and non-null counts
```bash
print(df.info())
```
output : 
```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6819 entries, 0 to 6818
Data columns (total 96 columns):
 #   Column                                                    Non-Null Count  Dtype  
---  ------                                                    --------------  -----  
 0    ROA(C) before interest and depreciation before interest  6819 non-null   float64
 1    ROA(A) before interest and % after tax                   6819 non-null   float64
 2    ROA(B) before interest and depreciation after tax        6819 non-null   float64
 3    Operating Gross Margin                                   6819 non-null   float64
 4    Realized Sales Gross Margin                              6819 non-null   float64
 5    Operating Profit Rate                                    6819 non-null   float64
 6    Pre-tax net Interest Rate                                6819 non-null   float64
 7    After-tax net Interest Rate                              6819 non-null   float64
 8    Non-industry income and expenditure/revenue              6819 non-null   float64
 9    Continuous interest rate (after tax)                     6819 non-null   float64
 10   Operating Expense Rate                                   6819 non-null   float64
 11   Research and development expense rate                    6819 non-null   float64
 12   Cash flow rate                                           6819 non-null   float64
 13   Interest-bearing debt interest rate                      6819 non-null   float64
 14   Tax rate (A)                                             6819 non-null   float64
 15   Net Value Per Share (B)                                  6819 non-null   float64
 16   Net Value Per Share (A)                                  6819 non-null   float64
 17   Net Value Per Share (C)                                  6819 non-null   float64
 18   Persistent EPS in the Last Four Seasons                  6819 non-null   float64
 19   Cash Flow Per Share                                      6819 non-null   float64
 20   Revenue Per Share (Yuan ¥)                               6819 non-null   float64
 21   Operating Profit Per Share (Yuan ¥)                      6819 non-null   float64
 22   Per Share Net profit before tax (Yuan ¥)                 6819 non-null   float64
 23   Realized Sales Gross Profit Growth Rate                  6819 non-null   float64
 24   Operating Profit Growth Rate                             6819 non-null   float64
 25   After-tax Net Profit Growth Rate                         6819 non-null   float64
 26   Regular Net Profit Growth Rate                           6819 non-null   float64
 27   Continuous Net Profit Growth Rate                        6819 non-null   float64
 28   Total Asset Growth Rate                                  6819 non-null   float64
 29   Net Value Growth Rate                                    6819 non-null   float64
 30   Total Asset Return Growth Rate Ratio                     6819 non-null   float64
 31   Cash Reinvestment %                                      6819 non-null   float64
 32   Current Ratio                                            6819 non-null   float64
 33   Quick Ratio                                              6819 non-null   float64
 34   Interest Expense Ratio                                   6819 non-null   float64
 35   Total debt/Total net worth                               6819 non-null   float64
 36   Debt ratio %                                             6819 non-null   float64
 37   Net worth/Assets                                         6819 non-null   float64
 38   Long-term fund suitability ratio (A)                     6819 non-null   float64
 39   Borrowing dependency                                     6819 non-null   float64
 40   Contingent liabilities/Net worth                         6819 non-null   float64
 41   Operating profit/Paid-in capital                         6819 non-null   float64
 42   Net profit before tax/Paid-in capital                    6819 non-null   float64
 43   Inventory and accounts receivable/Net value              6819 non-null   float64
 44   Total Asset Turnover                                     6819 non-null   float64
 45   Accounts Receivable Turnover                             6819 non-null   float64
 46   Average Collection Days                                  6819 non-null   float64
 47   Inventory Turnover Rate (times)                          6819 non-null   float64
 48   Fixed Assets Turnover Frequency                          6819 non-null   float64
 49   Net Worth Turnover Rate (times)                          6819 non-null   float64
 50   Revenue per person                                       6819 non-null   float64
 51   Operating profit per person                              6819 non-null   float64
 52   Allocation rate per person                               6819 non-null   float64
 53   Working Capital to Total Assets                          6819 non-null   float64
 54   Quick Assets/Total Assets                                6819 non-null   float64
 55   Current Assets/Total Assets                              6819 non-null   float64
 56   Cash/Total Assets                                        6819 non-null   float64
 57   Quick Assets/Current Liability                           6819 non-null   float64
 58   Cash/Current Liability                                   6819 non-null   float64
 59   Current Liability to Assets                              6819 non-null   float64
 60   Operating Funds to Liability                             6819 non-null   float64
 61   Inventory/Working Capital                                6819 non-null   float64
 62   Inventory/Current Liability                              6819 non-null   float64
 63   Current Liabilities/Liability                            6819 non-null   float64
 64   Working Capital/Equity                                   6819 non-null   float64
 65   Current Liabilities/Equity                               6819 non-null   float64
 66   Long-term Liability to Current Assets                    6819 non-null   float64
 67   Retained Earnings to Total Assets                        6819 non-null   float64
 68   Total income/Total expense                               6819 non-null   float64
 69   Total expense/Assets                                     6819 non-null   float64
 70   Current Asset Turnover Rate                              6819 non-null   float64
 71   Quick Asset Turnover Rate                                6819 non-null   float64
 72   Working capitcal Turnover Rate                           6819 non-null   float64
 73   Cash Turnover Rate                                       6819 non-null   float64
 74   Cash Flow to Sales                                       6819 non-null   float64
 75   Fixed Assets to Assets                                   6819 non-null   float64
 76   Current Liability to Liability                           6819 non-null   float64
 77   Current Liability to Equity                              6819 non-null   float64
 78   Equity to Long-term Liability                            6819 non-null   float64
 79   Cash Flow to Total Assets                                6819 non-null   float64
 80   Cash Flow to Liability                                   6819 non-null   float64
 81   CFO to Assets                                            6819 non-null   float64
 82   Cash Flow to Equity                                      6819 non-null   float64
 83   Current Liability to Current Assets                      6819 non-null   float64
 84   Liability-Assets Flag                                    6819 non-null   int64  
 85   Net Income to Total Assets                               6819 non-null   float64
 86   Total assets to GNP price                                6819 non-null   float64
 87   No-credit Interval                                       6819 non-null   float64
 88   Gross Profit to Sales                                    6819 non-null   float64
 89   Net Income to Stockholder's Equity                       6819 non-null   float64
 90   Liability to Equity                                      6819 non-null   float64
 91   Degree of Financial Leverage (DFL)                       6819 non-null   float64
 92   Interest Coverage Ratio (Interest expense to EBIT)       6819 non-null   float64
 93   Net Income Flag                                          6819 non-null   int64  
 94   Equity to Liability                                      6819 non-null   float64
 95  Bankrupt?                                                 6819 non-null   int64  
dtypes: float64(93), int64(3)
memory usage: 5.0 MB
None
```
#### Check for any missing values in the dataset
```bash
print(df.isnull().sum())
```
output : 
```bash
ROA(C) before interest and depreciation before interest    0
 ROA(A) before interest and % after tax                     0
 ROA(B) before interest and depreciation after tax          0
 Operating Gross Margin                                     0
 Realized Sales Gross Margin                                0
                                                           ..
 Degree of Financial Leverage (DFL)                         0
 Interest Coverage Ratio (Interest expense to EBIT)         0
 Net Income Flag                                            0
 Equity to Liability                                        0
Bankrupt?                                                   0
Length: 96, dtype: int64
```
## **Split the dataset into training and testing sets**
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
## **Feature scaling**
```bash
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## **Train the Logistic Regression model**
```bash
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```
## **Make predictions on the test set**
```bash
predictions = classifier.predict(X_test)
```
## **Print predictions alongside the actual values**
```bash
print(np.concatenate((predictions.reshape(len(predictions), 1), y_test.reshape(len(y_test), 1)), 1))
```
output :
```bash
[[0 0]
 [0 0]
 [0 0]
 ...
 [0 0]
 [0 0]
 [0 0]]
```
## **Cross-validation accuracy**
```bash
acc = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Cross-Validation Accuracy: {:.2f}%".format(acc.mean() * 100))
print("Standard Deviation: {:.2f}%".format(acc.std() * 100))
```
output :
```bash
Cross-Validation Accuracy: 96.55%
Standard Deviation: 0.38%
```
## **Confusion Matrix**
```bash
print(confusion_matrix(y_test, predictions))
```
output : 
```bash
[[1310    8]
 [  39    7]]
```
#### Plot the confusion matrix
```bash
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```
output :
![image](https://github.com/user-attachments/assets/62c1957e-2f04-4750-8d82-6be3d0af4a12)
## **Plot ROC Curve**
```bash
y_prob = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
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
output : 
![image](https://github.com/user-attachments/assets/8fc53f7e-8838-4d3b-adc8-781841fdd70b)
# Conclusion
### **In this project, we successfully built a logistic regression model to predict company bankruptcy using financial indicators. After preprocessing the data, including scaling the features, we trained the model and evaluated its performance. The model achieved a solid accuracy score and showed reliable performance through cross-validation. The ROC curve further confirmed the model's ability to distinguish between defaulting and non-defaulting companies. This predictive model can be a useful tool for early detection of financial distress in companies, enabling timely decision-making for investors and stakeholders.**


















