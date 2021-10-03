import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
from sklearn.metrics import mean_squared_error
from scripts.common import load_data, compute_capacity, Setup, plot_training_history, calculate_Z_scores, \
    max_min_scale_1D
from argparse import ArgumentParser

tf.compat.v1.disable_eager_execution()


def neuralnet_batch_predict_capacities(model, rho, sigma_sq, RIS_profiles):
    magnitudes = np.empty((RIS_profiles.shape[0], len(rho)))
    capacities = np.empty(RIS_profiles.shape[0])

    for i in range(RIS_profiles.shape[0]):
        mag_H_f_sq = model.predict(RIS_profiles[i, :].reshape((1, -1)))
        capacity = compute_capacity(mag_H_f_sq, rho, sigma_sq)
        magnitudes[i, :] = mag_H_f_sq
        capacities[i] = capacity

    return capacities







parser = ArgumentParser()
parser.add_argument("-qq", type=int, default=1, help="Reverberation time index (1 to 3)")
parser.add_argument("-ll", type=int, default=3, help="Perturber size index (1 to 3)")
parser.add_argument("-snr", type=float, default=15.0, help="Transmit SNR value in dB")
args = parser.parse_args()


setup = Setup(args.ll, args.qq, args.snr)
print("\n Running setup: "+setup.get_description(' ','=')+"\n")


RIS_configs_train, avg_squared_mag_response_train, RIS_configs_test, avg_squared_mag_response_test = load_data(setup, together=False)


sns.distplot(avg_squared_mag_response_train, label='Train')
sns.distplot(avg_squared_mag_response_test, label='Test')
plt.xlabel('$\mathbb{E}[|H(f)|^2]$')
plt.show()




X_train      = RIS_configs_train.astype(np.float32)
X_test       = RIS_configs_test.astype(np.float32)


y_train      = avg_squared_mag_response_train
y_test       = avg_squared_mag_response_test


X_train = X_train + np.random.normal(scale=0.01, size=X_train.shape)
training_data_loss_weights = calculate_Z_scores(y_train)
training_data_loss_weights = max_min_scale_1D(training_data_loss_weights)


random_id = str(np.random.randint(2**32))
#random_id = 4058011840
print()
filename = setup.get_model_filename(random_id)




reg   = 5e-8
p     = 0.0005
inp   = keras.layers.Input((X_train.shape[1],))
x     = keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg))(inp)
x     = keras.layers.Dropout(p)(x)
x     = keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
x     = keras.layers.Dropout(p)(x)
# x     = keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(reg))(x)
# x     = keras.layers.Dropout(p)(x)
# x     = keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(reg))(x)
# x     = keras.layers.Dropout(p)(x)

out    = keras.layers.Dense(y_train.shape[1], activation='linear', name='out')(x)

model = keras.Model(inp, out)

stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, verbose=1)
tqdm_callback    = TqdmCallback(verbose=0)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                                loss='mse')





history = model.fit(X_train,
                    y_train,
                    epochs=2500,
                    batch_size=80,
                    validation_data=(X_test, y_test),
                    sample_weight=training_data_loss_weights,
                    verbose=0,
                    callbacks=[tqdm_callback, stopping_callback],
          )
plot_training_history(history)





model.save(filename)
print(f'>>> Saved model to "{filename}".')








model = keras.models.load_model(setup.get_model_filename(random_id))




print()
y_pred_train = model.predict(X_train)
print("Train MSE: {:e}".format(mean_squared_error(y_train, y_pred_train)))

y_pred_test = model.predict(X_test)
print("Test MSE : {:e}".format(mean_squared_error(y_test, y_pred_test)))



train_capacities = np.empty(y_train.shape[1])
test_capacities  = np.empty(y_test.shape[1])

for i in range(y_train.shape[1]): train_capacities[i] = compute_capacity(y_train[i,:], setup.rho, setup.sigma_sq)

for i in range(y_test.shape[1]): test_capacities[i] = compute_capacity(y_test[i,:], setup.rho, setup.sigma_sq)

    
capacities_ground_truth    = np.concatenate([train_capacities, test_capacities])
xlim                       = [capacities_ground_truth.min() - capacities_ground_truth.std(), capacities_ground_truth.max()+capacities_ground_truth.std()]
random_RIS_profiles        = np.random.binomial(1, 0.5, size=(1000, X_train.shape[1])).astype(X_train.dtype)
capacities_rand_pred       = neuralnet_batch_predict_capacities(model, setup.rho, setup.sigma_sq, random_RIS_profiles)
random_continuous_RIS      = np.random.uniform(0, 1, size=(1000, X_train.shape[1]))
capacities_continuous_rand = neuralnet_batch_predict_capacities(model, setup.rho, setup.sigma_sq, random_continuous_RIS)


print()
print(f"data ground truth capacities                : mean {capacities_ground_truth.mean()}, std: {capacities_ground_truth.std()}")
#print(f"data predicted capacities                   : mean {capacities_data_pred.mean()}, std: {capacities_data_pred.std()}")
print(f"random data predicted capacities            : mean {capacities_rand_pred.mean()}, std: {capacities_rand_pred.std()}")
print(f"random data (continuous)predicted capacities: mean {capacities_continuous_rand.mean()}, std: {capacities_continuous_rand.std()}")





sns.set_theme()
fig, ax = plt.subplots(figsize=(15,8))

sns.distplot(capacities_ground_truth, hist=True, rug=False, label='Dataset (ground truth)')
#sns.distplot(capacities_data_pred,    hist=True, rug=False, label='Dataset (predicted)')
sns.distplot(capacities_rand_pred,    hist=True, rug=False, label='Random RIS profiles (predicted)')
sns.distplot(capacities_continuous_rand,    hist=True, rug=False, label='Continuous Random RIS profiles (predicted)')

plt.xlim(xlim)
plt.legend(fontsize=16)
plt.xlabel('$C(\Phi)$', fontsize=16)
plt.ylabel('Density', fontsize=16)
#plt.grid()
plt.savefig(filename+'.png')
plt.show()


sns.set_theme()
fig, ax = plt.subplots(figsize=(15,8))
capacities_pred_train      = neuralnet_batch_predict_capacities(model, setup.rho, setup.sigma_sq, X_train)
capacities_pred_test       = neuralnet_batch_predict_capacities(model, setup.rho, setup.sigma_sq, X_test)
capacities_data_pred       = np.concatenate([capacities_pred_train, capacities_pred_test])
sns.distplot(capacities_ground_truth, hist=True, rug=True, label='Dataset (ground truth)')
sns.distplot(capacities_data_pred, hist=True, rug=True, label='Dataset (predictions)')
plt.xlim(xlim)
plt.legend(fontsize=16)
plt.xlabel('$C(\Phi)$', fontsize=16)
plt.ylabel('Density', fontsize=16)
#plt.grid()
plt.savefig(filename+'.png')
plt.show()






print("\nID:", random_id)










