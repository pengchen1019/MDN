from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
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

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 12})

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

def remove_ax_window(ax):
    """
        Remove all axes and tick params in pyplot.
        Input: ax object.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis=u'both', which=u'both', length=0)

class MDN(tf.keras.Model):

    def __init__(self, neurons=100, components=2):
        super(MDN, self).__init__(name="MDN")
        self.neurons = neurons
        self.components = components

        self.h1 = Dense(neurons, activation="relu", name="h1")
        self.h2 = Dense(neurons, activation="relu", name="h2")

        self.alphas = Dense(components, activation="softmax", name="alphas")
        self.mus = Dense(components, name="mus")
        self.sigmas = Dense(components, activation="nnelu", name="sigmas")
        self.pvec = Concatenate(name="pvec")

    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)

        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)

        return self.pvec([alpha_v, mu_v, sigma_v])


class DNN(tf.keras.Model):
    def __init__(self, neurons=100):
        super(DNN, self).__init__(name="DNN")
        self.neurons = neurons

        self.h1 = Dense(neurons, activation="relu", name="h1")
        self.h2 = Dense(neurons, activation="relu", name="h2")
        self.out = Dense(1, activation="linear", name="out")

    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        return self.out(x)


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector)  # Unpack parameter vectors

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

    return -tf.reduce_mean(log_likelihood, axis=-1)

tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

no_parameters = 3
components = 2
neurons = 200

opt = tf.optimizers.Adam(1e-3)

mon = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

x_data = data[['xx_BPT_WHAN','yy_BPT','yy_WHAN']]
y_data = labels2.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)

# min_max_scaler = MinMaxScaler()
# x_train = min_max_scaler(x_train)
# x_test = min_max_scaler(x_test)

mdn_2 = MDN(neurons=neurons, components=components)
mdn_2.compile(loss=gnll_loss, optimizer=opt)

mdn_2.fit(x=x_train, y=y_train,epochs=200, validation_data=(x_test, y_test), callbacks=[mon], batch_size=128, verbose=0)
y_pred = mdn_2.predict(x_data)
alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred)

print(y_pred)


set_plt_style()
plt.show()