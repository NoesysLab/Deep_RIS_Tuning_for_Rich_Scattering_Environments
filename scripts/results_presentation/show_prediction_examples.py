import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import itertools
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns


from scripts.common import compute_capacity, Setup, load_data

import tensorflow as tf

from scripts.genetic_algorithm import genetic_algorithm

tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K


random_id = 1501880932


setup          = Setup(1, 3, 15)
model_filename = setup.get_model_filename(random_id)
model          = tf.keras.models.load_model(model_filename)



import visualkeras
from PIL import ImageFont
#font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, legend=True,  to_file='output.png').show()  # font is optional!








RIS_configs_train, avg_squared_mag_response_train, RIS_configs_test, avg_squared_mag_response_test = load_data(setup, together=False)

#
# sns.distplot(avg_squared_mag_response_train, label='Train')
# sns.distplot(avg_squared_mag_response_test, label='Test')
# plt.xlabel('$\mathbb{E}[|H(f)|^2]$')
# plt.show()




X_train      = RIS_configs_train.astype(np.float32)
X_test       = RIS_configs_test.astype(np.float32)


y_train      = avg_squared_mag_response_train
y_test       = avg_squared_mag_response_test


X_train = X_train + np.random.normal(scale=0.01, size=X_train.shape)