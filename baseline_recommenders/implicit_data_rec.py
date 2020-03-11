import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# # Drop NaN columns
# data = raw_data.dropna()
# data = data.copy()
#
# # Create a numeric user_id and brand_id column
def create_sparse_matrix(data):

    data['user_id'] = data['user_id'].astype("category")
    data['brand'] = data['brand'].astype("category")
    data['user_id'] = data['user_id'].cat.codes
    data['brand_id'] = data['brand'].cat.codes
    print(data)

    # Create a lookup frame so we can get the brand names back in
    # readable form later.
    item_lookup = data[['brand_id', 'brand']].drop_duplicates()
    item_lookup['brand_id'] = item_lookup.brand_id.astype(str)
    print(item_lookup)
    # user_lookup = data[['user_id', 'user']].drop_duplicates()
    # user_lookup['user_id'] = item_lookup.mcat_id.astype(str)
    #
    data = data.drop(['brand','event_type'], axis=1)
    print(data)
    # # Drop any rows that have 0 purchases
    # data = data.loc[data.purchase_cnt != 0]

    # Create lists of all users, mcats and their purchase counts
    users = list(np.sort(data.user_id.unique()))
    brands = list(np.sort(data.brand_id.unique()))
    actions = list(data.rating)

    #print(users,brands,actions)
    # Get the rows and columns for our new matrix
    rows = data.user_id.astype(int)
    cols = data.brand_id.astype(int)
    #print(rows,cols)
    # Create a sparse matrix for our users and mcats containing number of purchases
    data_sparse_new = csr_matrix((actions, (rows, cols)), shape=(len(users), len(brands)))

    return data_sparse_new




