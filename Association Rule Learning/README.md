# Project Overview

#### In this project, we will perform a Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from a grocery store dataset.

#### We will start by loading the dataset and inspecting its structure, including checking for data types and understanding the contents.

#### We'll filter and sort the data to focus on relevant fields, such as Member_number and itemDescription, which represent customer transactions.

#### We'll group the items purchased by each member into lists, making it easier to analyze patterns in their purchasing behavior.

#### We'll apply the Apriori algorithm to find frequent itemsets and generate association rules based on specified thresholds for support, confidence, and lift.

#### Finally, we'll extract the association rules, format them into a readable structure, and sort them by their lift to highlight the most significant associations.

#### This analysis will help uncover patterns in customer purchases, which can be used to inform product placements, promotions, and inventory management strategies.

# First, try doing it on your own. If you struggle with something, you can find the steps outlined below.

## **First we install apriori**
```bash
!pip install apyori
```
## **Import necessary Libraries**
```bash
import pandas as pd
from apyori import apriori
```
## **Load and explore the data**
##### We start by loading the grocery dataset and examining the first few rows and the data type to understand the structure and content of the data.
```bash
df = pd.read_csv('Groceries data.csv')
```
##### Display the first few rows of the data
```bash
print(df.head())
```
output :
```bash
 Member_number   itemDescription
0           1808    tropical fruit
1           2552        whole milk
2           2300         pip fruit
3           1187  other vegetables
4           3037        whole milk
```
##### Display the data types of each column
```bash
print(df.info())
```
output :
```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 38765 entries, 0 to 38764
Data columns (total 2 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   Member_number    38765 non-null  int64 
 1   itemDescription  38765 non-null  object
dtypes: int64(1), object(1)
memory usage: 605.8+ KB
None
```
## **Prepare the data for Apriori analysis**
##### We will sort the data by 'Member_number' in descending order
```bash
req_stuff = df[['Member_number', 'itemDescription']].sort_values(by='Member_number', ascending=False)
print(req_stuff)
```
output :
```bash
      Member_number        itemDescription
3578            5000                   soda
34885           5000    semi-finished bread
11728           5000  fruit/vegetable juice
9340            5000           bottled beer
19727           5000        root vegetables
...              ...                    ...
13331           1000             whole milk
17778           1000     pickled vegetables
6388            1000                sausage
20992           1000    semi-finished bread
8395            1000             whole milk

[38765 rows x 2 columns]
```
## **Group items by each member**
##### We will use the Apriori algorithm to find frequent itemsets and generate association rules based on the specified minimum support, confidence, lift, and rule length.
```bash
rules = apriori(transactions=member_list, min_support=0.002, min_confidence=0.0002, min_lift=3, min_length=2, max_length=2)
```
##### Convert the rules into a list for easier handling and display the raw results
```bash
results = list(rules)
print(results)
```
output :
```bash
[RelationRecord(items=frozenset({'UHT-milk', 'kitchen towels'}), support=0.002308876346844536, ordered_statistics=[OrderedStatistic(items_base=frozenset({'UHT-milk'}), items_add=frozenset({'kitchen towels'}), confidence=0.029411764705882356, lift=3.821568627450981), OrderedStatistic(items_base=frozenset({'kitchen towels'}), items_add=frozenset({'UHT-milk'}), confidence=0.30000000000000004, lift=3.821568627450981)]), RelationRecord(items=frozenset({'beef', 'potato products'}), support=0.002565418163160595, ordered_statistics=[OrderedStatistic(items_base=frozenset({'beef'}), items_add=frozenset({'potato products'}), confidence=0.02145922746781116, lift=3.8021849395239955), OrderedStatistic(items_base=frozenset({'potato products'}), items_add=frozenset({'beef'}), confidence=0.4545454545454546, lift=3.8021849395239955)]), RelationRecord(items=frozenset({'canned fruit', 'coffee'}), support=0.002308876346844536, ordered_statistics=[OrderedStatistic(items_base=frozenset({'canned fruit'}), items_add=frozenset({'coffee'}), confidence=0.4285714285714286, lift=3.7289540816326534), OrderedStatistic(items_base=frozenset({'coffee'}), items_add=frozenset({'canned fruit'}), confidence=0.020089285714285716, lift=3.7289540816326534)]), RelationRecord(items=frozenset({'meat spreads', 'domestic eggs'}), support=0.0035915854284248334, ordered_statistics=[OrderedStatistic(items_base=frozenset({'domestic eggs'}), items_add=frozenset({'meat spreads'}), confidence=0.02697495183044316, lift=3.0042389210019267), OrderedStatistic(items_base=frozenset({'meat spreads'}), items_add=frozenset({'domestic eggs'}), confidence=0.4, lift=3.0042389210019267)]), RelationRecord(items=frozenset({'flour', 'mayonnaise'}), support=0.002308876346844536, ordered_statistics=[OrderedStatistic(items_base=frozenset({'flour'}), items_add=frozenset({'mayonnaise'}), confidence=0.06338028169014086, lift=3.3385991625428253), OrderedStatistic(items_base=frozenset({'mayonnaise'}), items_add=frozenset({'flour'}), confidence=0.12162162162162163, lift=3.338599162542825)]), RelationRecord(items=frozenset({'rice', 'napkins'}), support=0.0030785017957927143, ordered_statistics=[OrderedStatistic(items_base=frozenset({'napkins'}), items_add=frozenset({'rice'}), confidence=0.03785488958990536, lift=3.0113950943153287), OrderedStatistic(items_base=frozenset({'rice'}), items_add=frozenset({'napkins'}), confidence=0.2448979591836735, lift=3.011395094315329)]), RelationRecord(items=frozenset({'waffles', 'sparkling wine'}), support=0.002565418163160595, ordered_statistics=[OrderedStatistic(items_base=frozenset({'sparkling wine'}), items_add=frozenset({'waffles'}), confidence=0.21739130434782608, lift=3.1501535477614353), OrderedStatistic(items_base=frozenset({'waffles'}), items_add=frozenset({'sparkling wine'}), confidence=0.03717472118959108, lift=3.1501535477614353)])]
```
## **Make the output more readable**
#####  We will extract the left-hand side (lhs) and right-hand side (rhs) of the rules, along with their support, confidence, and lift values, and store them in a DataFrame for better readability.
```bash
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(resultsinDataFrame)
```
output :
```bash
  Left Hand Side  Right Hand Side   Support  Confidence      Lift
0        UHT-milk   kitchen towels  0.002309    0.029412  3.821569
1            beef  potato products  0.002565    0.021459  3.802185
2    canned fruit           coffee  0.002309    0.428571  3.728954
3   domestic eggs     meat spreads  0.003592    0.026975  3.004239
4           flour       mayonnaise  0.002309    0.063380  3.338599
5         napkins             rice  0.003079    0.037855  3.011395
6  sparkling wine          waffles  0.002565    0.217391  3.150154
```
## **Sort and display the top rules by lift**
##### Finally, we will sort the association rules by their lift values in descending order and display the top 10 rules. This will highlight the strongest associations found in the dataset.
```bash
print(resultsinDataFrame.nlargest(n=10, columns='Lift'))
```
output :
```bash
Left Hand Side  Right Hand Side   Support  Confidence      Lift
0        UHT-milk   kitchen towels  0.002309    0.029412  3.821569
1            beef  potato products  0.002565    0.021459  3.802185
2    canned fruit           coffee  0.002309    0.428571  3.728954
4           flour       mayonnaise  0.002309    0.063380  3.338599
6  sparkling wine          waffles  0.002565    0.217391  3.150154
5         napkins             rice  0.003079    0.037855  3.011395
3   domestic eggs     meat spreads  0.003592    0.026975  3.004239
```
# Conclusion
#### This project applied the Apriori algorithm to analyze grocery store transactions, revealing key patterns in customer purchasing behavior.
#### The identified association rules, particularly those with high lift values, offer actionable insights that can inform product placement and marketing strategies, ultimately enhancing the shopping experience and driving sales.



















