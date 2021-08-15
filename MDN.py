import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from tensorflow_probability import distributions as tfd

from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from gmm_fig_style import *
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("always")

plt.style.use('seaborn-whitegrid')
sns.set_palette(sns.color_palette('Set2', 4))
my_cmap = mpl.cm.colors.ListedColormap(sns.color_palette('Set2', 4),
                                       name='from_list', N=4)
print('Color palette used throughout the notebook :')
sns.palplot(sns.color_palette("Set2",4))
data = pd.read_csv('./data/GMM_catalogue.csv')

plt.figure(figsize=(16,6))

ax1 = plt.subplot(121)
data.plot.hexbin(x='xx_BPT_WHAN', y='yy_BPT', mincnt=1,
                 bins='log', gridsize=101, cmap='viridis', ax=ax1)

ax2 = plt.subplot(122)
data.plot.hexbin(x='xx_BPT_WHAN', y='yy_WHAN', mincnt=1,
                 bins='log', gridsize=101, cmap='viridis', ax=ax2)

plt.show()

gmm2 = GaussianMixture(covariance_type='full', n_components=2)
gmm2.fit(data[['xx_BPT_WHAN','yy_BPT','yy_WHAN']])

# return the probability of belonging to a group
proba_gmm2 = gmm2.predict_proba(data[['xx_BPT_WHAN','yy_BPT','yy_WHAN']])
labels2 = proba_gmm2.argmax(axis=1)

# add to the initial dataframe new columns
# containing the probability to belong to a group
data['gmm2_proba1'] = proba_gmm2[:,0]
data['gmm2_proba2'] = proba_gmm2[:,1]

x_data = data[['xx_BPT_WHAN','yy_BPT','yy_WHAN']]
y_data = labels2.astype(np.float)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)

NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo

def get_mixture_coef(output):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")

  out_pi, out_sigma, out_mu = tf.split(1, 3, output)

  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.sub(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  normalize_pi = tf.inv(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.mul(normalize_pi, out_pi)

  out_sigma = tf.exp(out_sigma)

  return out_pi, out_sigma, out_mu

out_pi, out_sigma, out_mu = get_mixture_coef(output)

oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
def tf_normal(y, mu, sigma):
  result = tf.sub(y, mu)
  result = tf.mul(result,tf.inv(sigma))
  result = -tf.square(result)/2
  return tf.mul(tf.exp(result),tf.inv(sigma))*oneDivSqrtTwoPI

def get_lossfunc(out_pi, out_sigma, out_mu, y):
  result = tf_normal(y, out_mu, out_sigma)
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)
  return tf.reduce_mean(result)

lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

NEPOCH = 10000
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})

plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
plt.show()

x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate != x):
      return i
  print('error with sampling ensemble')
  return -1

def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0

  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      idx = get_pi_idx(result[i, j], out_pi[i])
      mu = out_mu[i, idx]
      std = out_sigma[i, idx]
      result[i, j] = mu + rn[i, j]*std
  return result