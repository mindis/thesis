from apyori import apriori
import pandas as pd

def create_association_rules_df(data,userkey,itemkey):

    user_interactions_number = data[userkey].value_counts()
    user_interactions_number = pd.DataFrame(user_interactions_number)
    user_interactions_number = user_interactions_number.rename(columns={userkey: 'freq'})
    user_interactions_number.index.names = [userkey]
    user_interactions_number = user_interactions_number[user_interactions_number['freq']<=50]
    list = user_interactions_number.index.tolist()
    data = data[data[userkey].isin(list)]

    #user_interactions_number.
    product_values = data.groupby(userkey)[itemkey].unique().to_dict()
    product_values_df = pd.DataFrame(product_values.values())

    return product_values_df


if __name__ == '__main__':

    FILEPATH = '/home/nick/Desktop/thesis/datasets/pharmacy-data/ratings-data/ratings_idx.csv'

    data = pd.read_csv(FILEPATH)
    df = create_association_rules_df(data,userkey='user_id',itemkey='product_id')
    print(df)


    records = []
    for i in range(len(df)):
        records.append([str(df.values[i, j]) for j in range(0, 50)])

    print(records)


    print("association rule generation started...")


    association_rules = apriori(records, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
    print(association_rules)
    association_results = list(association_rules)
    print('Association rule generation completed.\n')
    association_results_df = pd.DataFrame(association_results)
    print(association_results_df)

    for i in range(0, len(association_results)):
        print(association_results[i][0])



    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])

        # second index of the inner list
        print("Support: " + str(item[1]))

        # third index of the list located at 0th
        # of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")


