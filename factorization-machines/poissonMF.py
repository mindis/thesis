import numpy as np
import pandas as pd
from hpfrec import HPF
from poismf import PoisMF
from scipy.sparse import coo_matrix


"""==================Hierarchical Poisson Model==================="""
def main():

    #df = pd.read_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/indexed_implicit_ratings.csv')
    df = pd.read_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/implicit-data/implicit_feedback_dataset.csv')
    df.rename(columns={"user_id": "UserId", "product_id": "ItemId","eventStrength":"Count"},inplace=True)
    print(df)

    "Map userids with user indexing"
    userids = df['UserId'].unique()
    useridmap = pd.Series(data=np.arange(len(userids)), index=userids)
    #model = PoisMF()
    recommender = HPF()
    recommender = HPF(
        k=30, a=0.3, a_prime=0.3, b_prime=1.0,
        c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
        stop_crit='train-llk', check_every=10, stop_thr=1e-3,
        users_per_batch=None, items_per_batch=None, step_size=lambda x: 1/np.sqrt(x+2),
        maxiter=200, reindex=True, verbose=True,
        random_seed = None, allow_inconsistent_math=False, full_llk=False,
        alloc_full_phi=False, keep_data=True, save_folder=None,
        produce_dicts=True, keep_all_objs=True, sum_exp_trick=False)


    recommender.fit(df, val_set=df.sample(10**4))

    userid = 44

    """Get Top-N item recommendations for {userid} """
    print(recommender.topN(user=useridmap.index[userid], n=10, exclude_seen=True))

    return


if __name__ == "__main__":


    main()
