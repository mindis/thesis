"""
This module illustrates how to retrieve the top-10 items with highest rating
prediction. We first train an SVD algorithm on the MovieLens dataset, and then
predict all the ratings for the pairs (user, item) that are not in the training
set. We then retrieve the top-10 prediction for each user.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict


from surprise import SVD , SVDpp,SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans,\
                  KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise import accuracy
import pandas as pd
import numpy as np


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

if __name__ == '__main__':

    file_path = '/home/nick/Desktop/thesis/datasets/pharmacy-data/ratings-data/user_product_ratings_idx_50k.csv'
    reader = Reader(line_format='user item rating', sep=',',skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)


    for algorithm in [BaselineOnly()]: #SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(),
    #               KNNWithZScore(), BaselineOnly(), CoClustering()]:


        trainset = data.build_full_trainset()
        algorithm.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algorithm.test(testset)
        #print(predictions)
        print('{0}\nRMSE: {1:.2f}'.format(algorithm,accuracy.rmse(predictions)))


    # algo = SVDpp()
# trainset = data.build_full_trainset()
# algo.fit(trainset)
# testset = trainset.build_anti_testset()
# predictions = algo.test(testset)
# print(predictions)
# print('RMSE: {0:.2f}'.format(accuracy.rmse(predictions)))
# for trainset, testset in kf.split(data):
#
#     # train and test algorithm.
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#
#     # Compute and print Root Mean Squared Error
#     accuracy.rmse(predictions, verbose=True)
# Than predict ratings for all pairs (u, i) that are NOT in the training set.

    top_n = get_top_n(predictions, n=10)
    # df = pd.read_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/indexed_ratings10k.csv')
    # print(df['user_id'].nunique())
    # #cross_validate(BaselineOnly(), data, verbose=True)
    topN_df = pd.DataFrame()
    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
         topN_df[uid] = np.array([iid for (iid, _) in user_ratings])

    print(topN_df)
