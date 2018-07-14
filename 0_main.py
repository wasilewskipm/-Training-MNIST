"""
MNIST dataset from Kaggle

Piotr Wasilewski, 9.07.2018
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


""" definitions """
def digit_plot(df):
    global PIXEL_ROOT
    # func for plotting digits
    if len(df.shape) != 1:
        return('Plot aborted: Provide 1D array')
    elif df.shape[0] != 785:
        return('Plot aborted: Provide 785 element array')
    else:
        plt.imshow(df[-PIXEL_ROOT**2:].values.reshape(PIXEL_ROOT, PIXEL_ROOT), cmap = 'gray')
        plt.title('True value: {}'.format(df[0].astype(int)))
        plt.axis('off')
        plt.show()
    
def next_batch(df, oh, size):
    # func for initializing random batch
    batch_idx = np.arange(df.shape[0])
    np.random.shuffle(batch_idx)
    batch_idx = batch_idx[:size]
    
    return(oh.iloc[batch_idx, :], df.iloc[batch_idx, 1:])
    
    
def plot_weights():
    # func to plot tensorflow weights
    global PIXEL_ROOT
    wght = session.run(weights)
    
    fig, axes = plt.subplots(2, 5)
    fig.tight_layout()
    fig.subplots_adjust(top = .5, bottom = 0)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(wght[:, i].reshape(PIXEL_ROOT, PIXEL_ROOT), vmin=wght.min(), vmax=wght.max(), cmap='seismic')
        
        ax.set_title("Weights: {0}".format(i))
        ax.axis('off')

    plt.show()

    
""" prep code """    
# load files
df_train = pd.read_csv('../Input/train.csv')
df_test = pd.read_csv('../Input/test.csv')
df_valid = df_train.iloc[-4200:, :]
df_train = df_train.iloc[:37801, :]


# find maximum 'color' value
pixel_max = df_train.iloc[:, 1:].values.max()
PIXEL_ROOT = int((df_train.shape[1] - 1) ** .5)

# print descriptions
print(df_train.head())
print('Number of NAs in train dataset: {}'.format(df_train.isna().values.sum()))
print('Number of NAs in test dataset: {}'.format(df_test.isna().values.sum()))
print('Maximal value from pixel columns: {}'.format(pixel_max))

# normalize and invert pixel values
df_train.iloc[:, 1:] = 1 - df_train.iloc[:, 1:]/pixel_max

# visualise random digit
digit_plot(df_train.iloc[7], PIXEL_ROOT)


""" model code """
# create one-hot vector
onehot_train = pd.get_dummies(df_train.iloc[:, 0])
onehot_valid = pd.get_dummies(df_valid.iloc[:, 0])

# model parameters
np.random.seed(7)
mdl_batchsize = 100
mdl_iterations = 20
mdl_learnrate = .2
mdl_accuracy = np.array([None]*mdl_iterations)

# tensorflow variables
features = tf.placeholder(tf.float32, [None, PIXEL_ROOT**2])
categories_onehot = tf.placeholder(tf.float32, [None, onehot_train.shape[1]])
categories_digit = tf.placeholder(tf.int64, [None])
weights = tf.Variable(tf.zeros([PIXEL_ROOT**2, onehot_train.shape[1]]))
biases = tf. Variable(tf.zeros([onehot_train.shape[1]]))
dict_valid = {features: df_valid.iloc[:, 1:], categories_onehot: onehot_valid, categories_digit: df_valid.iloc[:, 0]}

# tensorflow model
logits = tf.matmul(features, weights) + biases
categories_onehot_pred = tf.nn.softmax(logits)
categories_digit_pred = tf.argmax(categories_onehot_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=categories_onehot)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=mdl_learnrate).minimize(cost)

correct_prediction = tf.equal(categories_onehot, categories_onehot_pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run tensorflow
session = tf.Session()
session.run(tf.global_variables_initializer())

for ii in range(mdl_iterations):
    categories_onehot_batch, features_batch = next_batch(df_train, onehot_train, mdl_batchsize)
    session.run(optimizer, feed_dict = {features: features_batch, categories_onehot: categories_onehot_batch})
    
    mdl_accuracy[ii] = session.run(accuracy, feed_dict=dict_valid)*100
    print('Iteration: {0}, Accuracy: {1:.1f}%'.format(ii, mdl_accuracy[ii]))

plot_weights()
plt.plot(np.arange(mdl_iterations), mdl_accuracy)

session.close()