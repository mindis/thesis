import implicit
import pandas as pd
import numpy as np
from create_sparse_matrix import create_sparse_matrix
from scipy.sparse import csr_matrix
from implicit.evaluation import precision_at_k , ndcg_at_k



def train_test_split(ratings, train_percentage=0.8):
    """ Randomly splits the ratings matrix into two matrices for training/testing.
    Parameters
    ----------
    ratings : coo_matrix
        A sparse matrix to split
    train_percentage : float
        What percentage of ratings should be used for training
    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of csr_matrices for training/testing """

    ratings = ratings.tocoo()
    userids = ratings.row
    unique_elements, counts_elements = np.unique(userids, return_counts=True)
    print(unique_elements,counts_elements)
    train_freq = np.floor(counts_elements * train_percentage)
    train_freq = train_freq.astype(int)
    test_freq = counts_elements - train_freq
    print(counts_elements, train_freq, test_freq)

    train_index = []
    test_index = []
    #print(len(unique_elements))
    for i in range(len(unique_elements)):
        #print(i)
        #print(counts_elements[i])
        for j in range(counts_elements[i]):
            # print(counts_elements[i])
            # print(j)
            bool_val = j <= train_freq[i]-1
            num1 = lambda x: 1 if x==True else 0
            num2 = lambda x: 0 if x==True else 1
            train_index = np.append(train_index,num1(bool_val))
            test_index = np.append(test_index,num2(bool_val))
            #test_index = j > train_freq[i]-1

    train_index = train_index > 0
    test_index = test_index > 0

    # random_index = np.random.random(len(ratings.data))
    # train_index = random_index < train_percentage
    # test_index = random_index >= train_percentage

    print(train_index)

    train = csr_matrix((ratings.data[train_index],
                        (ratings.row[train_index], ratings.col[train_index])),
                       shape=ratings.shape, dtype=ratings.dtype)

    test = csr_matrix((ratings.data[test_index],
                       (ratings.row[test_index], ratings.col[test_index])),
                      shape=ratings.shape, dtype=ratings.dtype)

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return train, test

if __name__ == '__main__':

    PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_ratings.csv'
    data = pd.read_csv(PATH)
    userkey = 'user_id'
    itemkey = 'product_id'

    #data.drop(columns='brand',inplace=True)

    # sessions_num = data.groupby('user_id').count()
    # sessions_num = pd.DataFrame(sessions_num['product_id'])
    # sessions_num['train_num'] = np.floor(sessions_num['product_id'] * 0.8)
    # sessions_num['test_num'] = sessions_num['product_id'] - sessions_num['train_num']
    # sessions_num.rename(columns={"product_id": "session_num"},inplace=True)
    # sessions_num['train_num'] = sessions_num['train_num'].astype(int)
    # sessions_num['test_num'] = sessions_num['test_num'].astype(int)
    # #train_freq = train_freq.astype(int)
    # sessions_num.reset_index(inplace=True)
    # print(sessions_num)
    #
    # data['user_id'] = data.groupby('user_id').ngroup()
    # data['product_id'] = data.groupby('product_id').ngroup()
    #
    #
    # # train_data = pd.DataFrame()
    # # test_data = pd.DataFrame()
    # for i in range(len(sessions_num)):
    #
    #     if i == 0 :
    #         train = data[data['user_id'] == i][:sessions_num['train_num'][i]]
    #         test = data[data['user_id'] == i][sessions_num['train_num'][i]:]
    #         train_data = train
    #         test_data = test
    #     else:
    #         train = data[data['user_id'] == i][:sessions_num['train_num'][i]]
    #         test = data[data['user_id'] == i][sessions_num['train_num'][i]:]
    #
    #         train_data = pd.concat([train_data,train],ignore_index=True)
    #         test_data = pd.concat([test_data,test],ignore_index=True)
    #
    # #print(train_data,test_data)
    #
    #
    # train_csr, user_lookup1, item_lookup = create_sparse_matrix(data,userkey,itemkey)
    # test_csr, user_lookup, item_lookup = create_sparse_matrix(data,userkey,itemkey)
    csr_data, user_lookup, item_lookup = create_sparse_matrix(data,userkey,itemkey)
    #print(csr_data)

    csr_data = csr_data.T.tocsr()
    print(csr_data)
    train,test = train_test_split(csr_data)
    print(train,test)

    #print(user_lookup,item_lookup)

    """initialize a model --- choose a model"""
    #model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=50)
    model = implicit.als.AlternatingLeastSquares(factors=50)
    #model = implicit.bpr.BayesianPersonalizedRanking(factors=100)
    #model = implicit.lmf.LogisticMatrixFactorization(factors=100)
    #model = implicit.approximate_als.AnnoyAlternatingLeastSquares()
    print(train.T.tocsr())

    """Train the model on a sparse matrix of item/user/confidence weights"""
    model.fit(train.T.tocsr())

    """Evaluation Metrics Calculation"""
    precision = precision_at_k(model, train, test, K=20)
    ndcg = ndcg_at_k(model,train, test,K=20)

    print('Precision@20: {0}\n NDCG@20: {1}\n'.format(precision,ndcg))


    """Recommend N best items for each user"""
    top_rec_4all = model.recommend_all(test,N=20)
    top_rec_4all = top_rec_4all.T
    #top_rec_4all = pd.DataFrame(data=top_rec_4all,columns=user_lookup.index.categories)
    top_rec_4all = pd.DataFrame(data=top_rec_4all,columns=user_lookup.index.values)
    print(top_rec_4all)