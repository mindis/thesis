import numpy as np
import pandas as pd
#tf.placeholder only available in v1, so we have to work around.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import MinMaxScaler

PATH = '/home/nick/Desktop/thesis/datasets/pharmacy-data/ratings-data/user_product_ratings.csv'
df = pd.read_csv(PATH)
#df.drop(columns='Unnamed: 0',inplace=True)
print(df.shape)
#df = df[:50000]
#print(df.shape)


threshold = 30

user_interactions = pd.DataFrame(df['user_id'].value_counts())
user_interactions.index.name = 'user_id'
user_interactions.rename(columns=({'user_id':'freq'}),inplace=True)
user_interactions = user_interactions[user_interactions['freq'] >= threshold]

list = user_interactions.index.values
df = df[df['user_id'].isin(list)]
print(df.shape)

print('Number of unique users: ', df['user_id'].nunique())
print('Number of unique items: ', df['product_id'].nunique())
#print(df)

user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating')
print(user_item_matrix)
user_item_matrix.fillna(0, inplace=True)

users = user_item_matrix.index.tolist()
items = user_item_matrix.columns.tolist()

#user_item_matrix = user_item_matrix.as_matrix()
print(user_item_matrix.shape)

num_input = df['product_id'].nunique()
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}
#Build encoder/decoder model

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

#We will construct the model and the predictions

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op

y_true = X

#define loss function and optimizer, and minimize the squared error, and define the evaluation metrics

loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()

#Initialize the variables. Because TensorFlow uses computational graphs for its operations, placeholders and variables must be initialized.
init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()

with tf.Session() as session:
    epochs = 100
    batch_size = 50

    session.run(init)
    session.run(local_init)

    num_batches = int(user_item_matrix.shape[0] / batch_size)
    user_item_matrix = np.array_split(user_item_matrix, num_batches)

    for i in range(epochs):

        avg_cost = 0
        for batch in user_item_matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches

        print("epoch: {} Loss: {}".format(i + 1, avg_cost))

    user_item_matrix = np.concatenate(user_item_matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={X: user_item_matrix})

    pred_data = pred_data.append(pd.DataFrame(preds))

    pred_data = pred_data.stack().reset_index(name='rating')
    pred_data.columns = ['user_id', 'product_id', 'rating']
    pred_data['user_id'] = pred_data['user_id'].map(lambda value: users[value])
    pred_data['product_id'] = pred_data['product_id'].map(lambda value: items[value])

    keys = ['user_id', 'product_id']
    index_1 = pred_data.set_index(keys).index
    index_2 = df.set_index(keys).index

    top_ten_ranked = pred_data[~index_1.isin(index_2)]
    top_ten_ranked = top_ten_ranked.sort_values(['user_id', 'rating'], ascending=[True, False])
    top_ten_ranked = top_ten_ranked.groupby('user_id').head(10)
    print(top_ten_ranked)

    """Get Top-10 products for user_id []"""
    userID = input('Enter userID to generate recommendations:\n')
    print(f'Generating recommendations for userID {userID}...')
    top_ten_ranked = top_ten_ranked.loc[top_ten_ranked['user_id'] == userID]
    top_ten_products = pd.DataFrame(top_ten_ranked[['product_id','rating']])
    top_ten_products.reset_index(inplace=True)
    top_ten_products.drop(columns='index',inplace=True)
    print(top_ten_products)
