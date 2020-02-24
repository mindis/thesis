import pandas as pd
import numpy as np

from preprocess_data import clean_dataset

PATH = '/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/'
data = pd.read_csv(PATH + '2019-Oct.csv')  # 2019-Nov.csv for November records

# Take only lines with the brand specified
data_with_brand = data.dropna(subset=['brand'])

print(data_with_brand[['brand','event_type']])

total_brand_records = len(data_with_brand)
purchases_ratio = len(data_with_brand[data_with_brand['event_type']=='purchase']) / len(data_with_brand)
cart_ratio = len(data_with_brand[data_with_brand['event_type']=='cart']) / len(data_with_brand)
view_ratio = len(data_with_brand[data_with_brand['event_type']=='view']) / len(data_with_brand)
removefromcart_ratio = len(data_with_brand[data_with_brand['event_type']=='remove_from_cart']) / len(data_with_brand)

print('Purchase Ratio:{0:.2f}%\nCart Ratio:{1:.2f}%\nView Ratio:{2:.2f}%\n\
RemoveFromCart Ratio:{3:.2f}%\n'.format(purchases_ratio*100,cart_ratio*100,view_ratio*100,removefromcart_ratio*100))

print(data_with_brand['event_type'].value_counts())
print(data_with_brand['brand'].value_counts())

print('Percentage of Brand Records in Dataset:{0:.2f}%\n'.format(len(data_with_brand) / len(data) * 100))

df = clean_dataset(data_with_brand)
df = pd.DataFrame(df)
df.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/brand_dataset.csv')

brand_user_item = df.groupby(['user_id', 'product_id','brand']).event_type.value_counts().to_frame()
brand_user_item = pd.DataFrame(data=brand_user_item)
#user_item_matrix.to_csv(path_or_buf='/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/user_item_matrix.csv')
print(brand_user_item)
brand_user_item.to_csv('/home/nick/Desktop/thesis/datasets/cosmetics-shop-data/user_item_brand.csv')


# print(user_item_matrix)
# user_item_matrix.rename(columns={"event_type.1": "rating"}, inplace=True)
# user_item_matrix['rating'].astype(float)
# ratings = build_dataset_with_ratings(user_item_matrix, 100)
# print(ratings)