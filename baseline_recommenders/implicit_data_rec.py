import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.sparse import spdiags
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve



# # Drop NaN columns
# data = raw_data.dropna()
# data = data.copy()
#
# # Create a numeric user_id and brand_id column
def create_sparse_matrix(data,user_key = 'user_id',item_key='product_id'):

    data[user_key] = data[user_key].astype("category")
    data[item_key] = data[item_key].astype("category")
    #data['brand'] = data['brand'].astype("category")
    data['user'] = data[user_key].cat.codes
    data['item'] = data[item_key].cat.codes
    print(data)

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
    # user_lookup = data[['user_id', 'user']].drop_duplicates()
    # user_lookup['user_id'] = item_lookup.mcat_id.astype(str)
    #
    data = data.drop(['brand','event_type',user_key,item_key], axis=1)
    print(data)
    # # Drop any rows that have 0 purchases
    # data = data.loc[data.purchase_cnt != 0]

    # Create lists of all users, mcats and their purchase counts
    users = list(np.sort(data.user.unique()))
    items = list(np.sort(data.item.unique()))
    #brands = list(np.sort(data.brand_id.unique()))
    actions = list(data.rating)

    #print(users,brands,actions)
    # Get the rows and columns for our new matrix
    rows = data.user.astype(int)
    cols = data.item.astype(int)
    #print(rows,cols)
    # Create a sparse matrix for our users and brands containing number of purchases
    data_sparse_new = csr_matrix((actions, (cols, rows)), shape=(len(items), len(users)))

    return data_sparse_new, user_lookup, item_lookup


def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):
    # Calculate the Confidence for each value in our data
    confidence = sparse_data * alpha_val

    # Get the size of user rows and item columns using numpy array shape
    user_size, item_size = sparse_data.shape

    # We create the user vectors X of size users x features, the item vectors
    # Y of size items x features and randomly assign values to them using np.random.normal
    X = csr_matrix(np.random.normal(size=(user_size, features)))
    Y = csr_matrix(np.random.normal(size=(item_size, features)))

    # Identity matrix and lambda * I
    X_I = eye(user_size)
    Y_I = eye(item_size)

    I = eye(features)
    lI = lambda_val * I

    for i in range(iterations):
        print('iteration %d of %d' % (i + 1, iterations))

        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Run in a loop for entire user data
        for u in range(user_size):
            # Get the user row.
            u_row = confidence[u, :].toarray()

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

        for i in range(item_size):
            # Get the item column and transpose it.
            i_row = confidence[:, i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
            CiI = diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y

