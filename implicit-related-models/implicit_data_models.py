import implicit
import pandas as pd
import numpy as np
from create_sparse_matrix import create_sparse_matrix
from sklearn.preprocessing import MinMaxScaler
import time
from scipy.sparse import csr_matrix
from implicit.evaluation import precision_at_k , ndcg_at_k,train_test_split


def recommend(userID, user_item, user_vecs, item_vecs, num_items=10):

    # Get the interactions scores from the sparse person content matrix
    user_interactions = user_item[userID, :].toarray()
    # Add 1 to everything, so that articles with no interaction yet become equal to 1
    user_interactions = user_interactions.reshape(-1) + 1
    # Make articles already interacted zero
    user_interactions[user_interactions > 1] = 0
    # Get dot product of person vector and all content vectors
    rec_vector = user_vecs[userID, :].dot(item_vecs.T).toarray()
    print(rec_vector)

    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
    print(rec_vector_scaled)
    print(user_interactions)

    # Items already interacted have their recommendation multiplied by zero
    recommend_vector = user_interactions * rec_vector_scaled
    # Sort the indices of the items into order of best recommendations
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    # Start empty list to store titles and scores
    items = item_idx
    scores = []

    #print(item_idx)

    for idx in item_idx:
        # Append scores to the list
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'item': items, 'score': scores})

    return recommendations



if __name__ == "__main__":

    #PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_binarized.csv'
    PATH = '/home/nick/Desktop/thesis/datasets/pharmacy-data/ratings-data/user_product_ratings.csv'
    data = pd.read_csv(PATH)
    print(data)
    userkey = 'user_id'
    itemkey = 'product_id'


    item_user, user_lookup, item_lookup = create_sparse_matrix(data,userkey,itemkey)

    print(item_user)

    alpha_val = 15
    item_user = (item_user * alpha_val).astype('double')
    #print(item_user)

    """initialize a model --- choose a model"""
    #model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=50)
    model = implicit.als.AlternatingLeastSquares(factors=50)


    user_item = item_user.T.tocsr()
    model.fit(user_item)

    # """train/test split"""
    # train, test = train_test_split(user_item)
    # train = train.tocoo()
    # test = test.tocoo()
    # trainuserids = train.row
    # testuserids = test.row
    #
    # print(np.unique(trainuserids),np.unique(testuserids))

    user_vecs = csr_matrix(model.user_factors)
    item_vecs = csr_matrix(model.item_factors)

    # Create recommendations for user with id 2025
    user_id = 0

    recommendations = recommend(user_id, user_item, user_vecs, item_vecs)

    print(recommendations)


    # # train the model on a sparse matrix of item/user/confidence weights
    # start_time = time.time()
    # model.fit(train.T.tocsr())
    # #model.fit(csr_data)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    #
    # recommendations = model.recommend(0,test)
    # df = pd.DataFrame(recommendations,columns=['product_id','score'])
    # print(df)
    #
    #
    # """Find related items"""
    # itemID = 0
    # related = model.similar_items(itemID)
    # related = pd.DataFrame(related)
    # print("Related itemIDs:\n{}".format(related))
    #
    # """Calculate Precision@N & NDCG@N"""
    # precision = precision_at_k(model, train, test, K=20)
    # ndcg = ndcg_at_k(model, train, test, K=20)
    #
    # print('Precision@20: {0}\n NDCG@20: {1}\n'.format(precision, ndcg))
    #
    # """Recommend items to every user"""
    # top_rec_4all = model.recommend_all(test)
    # top_rec_4all = top_rec_4all.T
    # top_rec_4all = pd.DataFrame(data=top_rec_4all,columns=user_lookup.index.categories)
    # print(top_rec_4all)
    #
    # """Personalize predictions by selecting userID"""
    # preferred_userID = '000485159f123d9352d805458550f861'
    # userID = user_lookup.index.get_loc(preferred_userID)
    # recommendations = model.recommend(userID, test)
    # recommendations = pd.DataFrame(recommendations, columns=['item', 'score'])
    # recommendations['item'] = recommendations.item.astype(str)
    # """Merge indexed product with real product for user = userID"""
    # recommendations = recommendations.merge(item_lookup, on='item')
    # print('Top-10 Product recommendations for user {0} :\n {1}'.format(preferred_userID, recommendations))
    #
    #
    # # """Get the trained user and item vectors.
    # # We convert them to csr matrices
    # # and make recommendations"""
    # # user_vecs = csr_matrix(model.user_factors)
    # # item_vecs = csr_matrix(model.item_factors)
    # # # Create recommendations for person with id 50
    # # user_id = 2
    # #
    # # recommendations = recommend(user_id, user_item, user_vecs, item_vecs)
    # # recommendations['item'] = recommendations.item.astype(str)
    # # """Merge indexed product with real product for user = userID"""
    # # recommendations = recommendations.merge(item_lookup, on='item')
    # # print(recommendations)

