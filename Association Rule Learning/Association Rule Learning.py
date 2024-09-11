import pandas as pd
from apyori import apriori

# load the data
df = pd.read_csv('DATA/Groceries data.csv')

# check data
print(df.head())

# check data types
print(df.info())


# now we will use Apriori to find the best rules
# first we will take the stuff that we need
req_stuff = df[['Member_number', 'itemDescription']].sort_values(by='Member_number', ascending=False)
print(req_stuff)

# we will create a list to group each member items
member_list = [i[1]['itemDescription'].tolist() for i in list(req_stuff.groupby(['Member_number']))]

rules = apriori(transactions=member_list, min_support=0.002, min_confidence=0.0002, min_lift=3,  min_length=2,
                max_length=2)

# list the rules and store them in results
results = list(rules)

print(results)

# make the output more readable
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
# sort rules with to start with the largest lift
print(resultsinDataFrame.nlargest(n=10, columns='Lift'))

