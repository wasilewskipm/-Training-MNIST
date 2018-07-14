"""
MNIST dataset from Kaggle

Piotr Wasilewski, 9.07.2018

main file with core code
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def main():    
    """ prep code """    
    global PIXEL_ROOT
    
    # load files
    df_train = pd.read_csv('../Input/train.csv')
    df_test = pd.read_csv('../Input/test.csv')
    
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
    df_valid = df_train.iloc[-4200:, :]
    df_train = df_train.iloc[:37801, :]
    
    # visualise random digit
    digit_plot(df_train.iloc[7])
    digit_plot(df_valid.iloc[0])
    
    
    """ model code """
    # create one-hot vector
    onehot_train = pd.get_dummies(df_train.iloc[:, 0])
    onehot_valid = pd.get_dummies(df_valid.iloc[:, 0])
    
    # model parameters
    np.random.seed(7)
    mdl_batchsize = 100
    mdl_iterations = 2000
    mdl_learnrate = 0.1
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
    
    correct_prediction = tf.equal(categories_digit, categories_digit_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100
    
    # run tensorflow
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    for ii in range(mdl_iterations):
        categories_onehot_batch, features_batch = batch_pull(df_train, onehot_train, mdl_batchsize)
        session.run(optimizer, feed_dict = {features: features_batch, categories_onehot: categories_onehot_batch})
        
        mdl_accuracy[ii] = session.run(accuracy, feed_dict=dict_valid)
        print('Iteration: {0}, Accuracy: {1:.1f}%'.format(ii, mdl_accuracy[ii]))
    
    # plot final weights and accuracy per iteration
    weights_plot(session.run(weights))
    accuracy_plot(mdl_accuracy)
    
    # plot random error
    bln, pred = session.run([correct_prediction, categories_digit_pred], feed_dict=dict_valid)
    digit_plot(df_valid.iloc[~bln, :].iloc[0, :], pred[~bln][0])
    
    # finish session
    session.close()


""" definitions """
def digit_plot(df, pred = None):
    global PIXEL_ROOT
    # func for plotting digits
    if len(df.shape) != 1:
        return('Plot aborted: Provide 1D array')
    elif df.shape[0] != 785:
        return('Plot aborted: Provide 785 element array')
    else:
        if pred == None:
            plt.title('True value: {0}'.format(df[0].astype(int)))
        else:
            plt.title('True value: {0}, Pred value: {1}'.format(df[0].astype(int), pred))
            
        plt.imshow(df[-PIXEL_ROOT**2:].values.reshape(PIXEL_ROOT, PIXEL_ROOT), cmap = 'gray')
        plt.axis('off')
        plt.show()

    
def batch_pull(df, oh, size):
    # func for initializing random batch
    batch_idx = np.arange(df.shape[0])
    np.random.shuffle(batch_idx)
    batch_idx = batch_idx[:size]
    
    return(oh.iloc[batch_idx, :], df.iloc[batch_idx, 1:])
    
    
def weights_plot(wght):
    # func to plot tensorflow weights
    global PIXEL_ROOT
    
    fig = plt.figure()
    fig.suptitle('Model weights')
    fig.tight_layout()
    
    for ii in range(10):
        ax = fig.add_subplot(2, 5, ii+1)        
        ax.imshow(wght[:, ii].reshape(PIXEL_ROOT, PIXEL_ROOT), vmin=wght.min(), vmax=wght.max(), cmap='seismic')
        
        ax.set_title("{0}".format(ii))
        ax.axis('off')
        
    fig.suptitle('Model weights')        
    plt.show()


def accuracy_plot(arr):
    # func to plot accuracy per iteration
    plt.title('Accuracy based on the validation set')
    plt.xlabel('Iteration')
    plt.ylabel('% Accuracy')
    plt.plot(np.arange(len(arr)), arr)
    plt.show()
    
    
if __name__ == '__main__':
    main()