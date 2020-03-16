import pandas as pd
import numpy as np
import lightfm
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


if __name__ == "__main__":

    #TRAIN_PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_feedback_dataset.csv'
    TRAIN_PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_ratings.csv'
    #TEST_PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_feedback_testdata.csv'

    traindata = pd.read_csv(TRAIN_PATH)
    #testdata = pd.read_csv(TEST_PATH)
    print(traindata)
    print('\n')
    #print(testdata)
    user_key = 'user_id'
    item_key = 'product_id'
    csr_data1, user_lookup1, item_lookup1 = create_sparse_matrix(traindata,user_key,item_key)
    #csr_data2, user_lookup2, item_lookup2 = create_sparse_matrix(testdata,user_key,item_key)


    user_items_train = csr_data1.T.tocsr()
    #user_items_test = csr_data2.T.tocsr()

    print(user_items_train)
    print('\n')
    #print(user_items_test)
    #print('\n')
    print(user_items_train.shape)
    #print(user_items_test.shape)


    train,test = cross_validation.random_train_test_split(user_items_train)
    # print(train,test)
    # print(train.shape(),test.shape())

    model1 = LightFM(learning_rate=0.05, loss='bpr')
    model2 = LightFM(learning_rate=0.05, loss='warp')
    model1.fit(train,epochs=10)
    model2.fit(train,epochs=10)
    #ranks = model.predict(user_items_train,num_threads=1)
    #print(ranks)

    train_recall1_10 = recall_at_k(model1, train, k=10).mean()
    test_recall1_10 = recall_at_k(model1, test, k=10).mean()

    train_recall1_20 = recall_at_k(model1,train,k=20).mean()
    test_recall1_20 = recall_at_k(model1,test,k=20).mean()

    #train_mrr1 = reciprocal_rank(model1, train).mean()
    #train_mrr_20 = reciprocal_rank(model1, user_items_train).mean()
    #train_mrr2 = reciprocal_rank(model2, user_items_train).mean()

    train_recall2_10 = recall_at_k(model2, train, k=10).mean()
    test_recall2_10 = recall_at_k(model2, test, k=10).mean()

    train_recall2_20 = recall_at_k(model2, train, k=20).mean()
    test_recall2_20 = recall_at_k(model2, test, k=20).mean()

    #test_recall = recall_at_k(model, user_items_test, k=20).mean()
    print("BPR Train : Recall@10:{0:.3f}, Recall@20:{1:.3f}".format(train_recall1_10,train_recall1_20))
    print("BPR Test : Recall@10:{0:.3f}, Recall@20:{1:.3f}".format(test_recall1_10,test_recall1_20))

    #print("MRR:{0:.3f}".format(train_mrr1))

    print("WARP Train: Recall@10:{0:.3f}, Recall@20:{1:.3f}".format(train_recall2_10, train_recall2_20))
    print("WARP Test: Recall@10:{0:.3f}, Recall@20:{1:.3f}".format(test_recall2_10, test_recall2_20))

    #print("MRR:{0:.3f}".format(train_mrr2))

    #train_mrr = r(model, user_items_train).mean()
    #print(train_mrr)
    #
    #
    # print("Train MRR@20:{0:.3f}\n Test MRR@20:{1:.3f}\n".format(train_mrr,test_mrr))


