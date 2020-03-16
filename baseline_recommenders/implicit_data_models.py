import implicit
import pandas as pd
from create_sparse_matrix import create_sparse_matrix
from scipy.sparse import csr_matrix


if __name__ == "__main__":

    PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_ratings.csv'
    data = pd.read_csv(PATH)
    print(data)
    user_key = 'user_id'
    item_key = 'product_id'
    csr_data, user_lookup, item_lookup = create_sparse_matrix(data,user_key,item_key)
    print(csr_data)
    print(type(csr_data))

    """initialize a model"""
    model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=50)
    #model = implicit.als.AlternatingLeastSquares(factors=50)
    #model = implicit.bpr.BayesianPersonalizedRanking(factors=50)
    #model = implicit.lmf.LogisticMatrixFactorization(factors=100)
    #model = implicit.approximate_als.AnnoyAlternatingLeastSquares()

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(csr_data)
    # recommend items for a user
    user_items = csr_data.T.tocsr()
    print(user_items)


    top_rec_4all = model.recommend_all(user_items)
    top_rec_4all = top_rec_4all.T
    top_rec_4all = pd.DataFrame(data=top_rec_4all,columns=user_lookup.index.categories)
    print(top_rec_4all)

    """Personalize predictions by selecting userID"""
    preferred_userID = 12055855
    userID = user_lookup.index.get_loc(preferred_userID)
    recommendations = model.recommend(userID, user_items)
    recommendations = pd.DataFrame(recommendations, columns=['item', 'score'])
    recommendations['item'] = recommendations.item.astype(str)
    """Get Top-10 brands for user = userID"""
    recommendations = recommendations.merge(item_lookup, on='item')
    print('Top-10 Product recommendations for user {0} :\n {1}'.format(preferred_userID, recommendations))




