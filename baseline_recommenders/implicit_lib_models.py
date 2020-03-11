import implicit
import pandas as pd
from implicit_data_rec import create_sparse_matrix
from scipy.sparse import csr_matrix


if __name__ == "__main__":

    PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/user_item_brand.csv'
    data = pd.read_csv(PATH)
    print(data)
    csr_data = create_sparse_matrix(data)
    print(csr_data)

    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=50)
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(csr_data)
    # recommend items for a user
    user_items = csr_data.T.tocsr()
    print(user_items)
    recommendations = model.recommend(1, user_items)
    print(recommendations)
