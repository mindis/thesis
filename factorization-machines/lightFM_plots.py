import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lightfm import LightFM
from lightfm import cross_validation
from lightfm.datasets import fetch_movielens
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
from lightfm.evaluation import auc_score



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

    #data = data.drop(['brand','event_type',user_key,item_key], axis=1)
    #print(data)

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
    # Create a sparse matrix for our users and brands containing eventStrength values
    data_sparse_new = csr_matrix((actions, (cols, rows)), shape=(len(items), len(users)))

    return data_sparse_new, user_lookup, item_lookup


if __name__ == '__main__':


    #TRAIN_PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_ratings_thr5.csv'
    TRAIN_PATH = '/home/nick/Desktop/thesis/datasets/pharmacy-data/ratings-data/user_product_ratings.csv'
    # TEST_PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_feedback_testdata.csv'

    traindata = pd.read_csv(TRAIN_PATH)
    print(traindata)
    print('\n')
    # print(testdata)
    user_key = 'user_id'
    item_key = 'product_id'
    csr_data1, user_lookup1, item_lookup1 = create_sparse_matrix(traindata, user_key, item_key)

    user_items_train = csr_data1.T.tocsr()

    print(user_items_train)
    print('\n')

    print(user_items_train.shape)
    # print(user_items_test.shape)

    print("Splitting the data into train/test set...\n")
    train, test = cross_validation.random_train_test_split(user_items_train)

    alpha = 1e-05
    epochs = 50
    num_components = 32

    warp_model = LightFM(no_components=num_components,
                        loss='warp',
                        learning_schedule='adagrad',
                        max_sampled=100,
                        user_alpha=alpha,
                        item_alpha=alpha)

    bpr_model = LightFM(no_components=num_components,
                        loss='bpr',
                        learning_schedule='adagrad',
                        user_alpha=alpha,
                        item_alpha=alpha)

    warp_duration = []
    bpr_duration = []
    warp_auc = []
    bpr_auc = []

    print("Start Training...\n")

    for epoch in range(epochs):
        start = time.time()
        warp_model.fit_partial(train, epochs=1)
        warp_duration.append(time.time() - start)
        warp_auc.append(auc_score(warp_model, test, train_interactions=train).mean())

    for epoch in range(epochs):
        start = time.time()
        bpr_model.fit_partial(train, epochs=1)
        bpr_duration.append(time.time() - start)
        bpr_auc.append(auc_score(bpr_model, test, train_interactions=train).mean())


    x = np.arange(epochs)
    plt.plot(x, np.array(warp_auc))
    plt.plot(x, np.array(bpr_auc))
    plt.legend(['WARP AUC', 'BPR AUC'], loc='upper right')
    plt.show()

