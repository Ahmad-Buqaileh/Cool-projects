import pandas as pd
from apyori import apriori

df = pd.read_csv('DATA/Groceries data.csv')

print(df.head())

print(df.info())

req_stuff = df[['Member_number', 'itemDescription']].sort_values(by='Member_number', ascending=False)
print(req_stuff)

member_list = [i[1]['itemDescription'].tolist() for i in list(req_stuff.groupby(['Member_number']))]

rules = apriori(transactions=member_list, min_support=0.002, min_confidence=0.0002, min_lift=3,  min_length=2,
                max_length=2)

results = list(rules)

print(results)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


resultsinDataFrame = pd.DataFrame(inspect(results),
                                  columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(resultsinDataFrame)
print("\n\n\n")
print(resultsinDataFrame.nlargest(n=10, columns='Lift'))

