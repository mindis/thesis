import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.sparse import spdiags
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def create_sparse_matrix(data,user_key = 'user_id',item_key='product_id'):

    data[user_key] = data[user_key].astype("category")
    data[item_key] = data[item_key].astype("category")
    #data['brand'] = data['brand'].astype("category")
    data['user'] = data[user_key].cat.codes
    data['item'] = data[item_key].cat.codes
    #print(data)

    # Create a lookup frame so we can get the brand names back in
    # readable form later.
    user_lookup = data[['user', user_key]].drop_duplicates()
    item_lookup = data[['item', item_key]].drop_duplicates()
    #brand_lookup['brand_id'] = item_lookup.brand_id.astype(str)
    user_lookup['user'] = user_lookup.user.astype(str)
    user_lookup = pd.DataFrame(user_lookup)
    user_lookup.set_index(user_key,inplace=True)
    item_lookup['item'] = item_lookup.item.astype(str)
    print(user_lookup,item_lookup)

    data = data.drop([user_key,item_key], axis=1)
    print(data)
    # # Drop any rows that have 0 purchases
    # data = data.loc[data.purchase_cnt != 0]

    # Create lists of all users, items and their event_strength values
    users = list(np.sort(data.user.unique()))
    items = list(np.sort(data.item.unique()))
    #brands = list(np.sort(data.brand_id.unique()))
    #actions = list(data.eventStrength)
    actions = list(data.rating)

    #print(users,brands,actions)
    # Get the rows and columns for our new matrix
    rows = data.user.astype(int)
    cols = data.item.astype(int)
    #print(rows,cols)
    # Create a sparse matrix for our users and brands containing eventStrength values
    data_sparse_new = csr_matrix((actions, (cols, rows)), shape=(len(items), len(users)))

    return data_sparse_new, user_lookup, item_lookup

