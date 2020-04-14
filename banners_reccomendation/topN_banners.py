import pandas as pd
import numpy as np

def get_top_banners(userID,topN_products_df,bannerIDs,banner_product_df,N=10):

    product_list = topN_products_df[userID]

    rows_list = []
    for i in range(banner_product_df['banner_id'].nunique()):

        banner_product_list = banner_product_df[banner_product_df['banner_id'] == bannerIDs[i]]
        arr1 = np.array(banner_product_list['product_id'])
        arr2 = np.array(product_list)
        common_products = list(set(arr1).intersection(set(arr2)))
        dict = {}
        dict.update({'banner_id': banner_ids[i], 'product_included': len(common_products)})
        # top_banners.append({'banner_id':banner_ids[i], 'product_included': len(common_products)},ignore_index=True)
        rows_list.append(dict)

    top_banners = pd.DataFrame(rows_list)
    top_banners.sort_values(by='product_included', ascending=False, inplace=True)
    top_banners = top_banners.head(N)

    return top_banners




if __name__ == '__main__':

    file_path = '/home/nick/Desktop/thesis/datasets/pharmacy-data/topN_productids.csv'
    topN_products_df = pd.read_csv(file_path)
    topN_products_df.drop(columns='Unnamed: 0',inplace=True)
    print(topN_products_df)

    file_path2 = '/home/nick/Desktop/thesis/datasets/pharmacy-data/initial data/banners_products.csv'
    banner_product_df = pd.read_csv(file_path2)
    banner_ids = pd.Series(banner_product_df['banner_id'].unique())
    print(banner_ids)

    """userID to get top-N banner recommendations"""
    test_user_id = '00d9b6049ba0d1da0917b9c0d7292a9e'

    """call function to generate recommendations"""
    topN_banners_df = get_top_banners(test_user_id,topN_products_df,banner_ids,banner_product_df)

    print(topN_banners_df)

# banner_product_list = banner_product_df[banner_product_df['banner_id']==74]
# arr1 = np.array(banner_product_list['product_id'])
# arr2 = np.array(product_list)
# print(arr1,arr2)
# #common_products = np.intersect1d(np.array(banner_product_list['product_id']),np.array(product_list))
# #common_products = list(set(arr1) & set(arr2))
# common_products = list(set(arr1).intersection(set(arr1)))
# print(common_products)

#top_banners = pd.DataFrame(columns=['banner_id','product_included'])
