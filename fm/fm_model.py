from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


def vectorize(lil, ix=None, p=None):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    n -- number of samples
    g -- number of groups
    """
    if ix==None:
        ix = defaultdict(count(0).next)
    n = len(lil[0]) # num samples
    g = len(lil) # num groups
    nz = n * g

    col_ix = np.empty(nz,dtype = int)

    for i, d in enumerate(lil):
        # append index k with __i in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(k) + '__' + str(i)] for k in d]

    row_ix = np.repeat(np.arange(0,n),g)
    data = np.ones(nz)

    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

def main():
    cols = ['user','item','rating','timestamp']

    train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
    test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)

    x_train,ix = vectorize([train['user'].values,
                                train['item'].values])

    x_test,ix = vectorize([test['user'].values,
                            test['item'].values],ix,x_train.shape[1])


    y_train = train['rating'].values
    y_test = test['rating'].values

    x_train = x_train.todense()
    x_test = x_test.todense()

    #n is sample number, p is the one-hot coded metrics
    n, p = x_train.shape

    k = 10 # number of latent factors

    x = tf.placeholder('float',[None, p])

    y = tf.placeholder('float',[None, 1])

    w0 = tf.Variable(tf.zeros([1]))
    w = tf.Variable(tf.zeros([p]))

    v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))
    
    # estimate of y
    y_hat = tf.Variable(tf.zeros([n,1]))
    linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keepdims=True)) # w*x + w0
    pair_interactions = tf.multiply(0.5, tf.reduce_sum(
        tf.subtract(
            tf.pow(
                tf.matmul(x,tf.transpose(v)),2),
            tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
        ), axis = 1 , keepdims=True))


    y_hat = tf.add(linear_terms,pair_interactions)

    #loss function
    lambda_w = tf.constant(0.001,name='lambda_w')
    lambda_v = tf.constant(0.001,name='lambda_v')

    # reduce sum: sum across dimensions
    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w,tf.pow(w,2)),
            tf.multiply(lambda_v,tf.pow(v,2))
        )
    )

    error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
    loss = tf.add(error,l2_norm)

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    epochs = 10
    batch_size = 1000

    # Launch the graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in tqdm(range(epochs), unit='epoch'):
            perm = np.random.permutation(x_train.shape[0])
            # iterate over batches
            for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
                _, t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})


        errors = []
        for bX, bY in batcher(x_test, y_test):
            errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
            print("errors", errors)
        RMSE = np.sqrt(np.array(errors).mean())
        print (RMSE)
        sess.close()

if __name__ == "__main__":
    main()