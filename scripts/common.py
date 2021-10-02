import warnings
from dataclasses import dataclass, field

from scripts.genetic_algorithm import genetic_algorithm

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.io
import matplotlib.colors
import itertools
from time import sleep


import tensorflow as tf
from tensorflow import keras

import datetime
from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


DATA_DIR        = './data/summary/'
FILENAME_FORMAT = DATA_DIR + 'SUMMARY qq={:d} ll={:d}.mat'
QQ_VALUES       = [1,2,3]
LL_VALUES       = [1,2,3]
dB2power = lambda x: 10*np.log10(x/1.)






@dataclass
class Setup:
    LL               = 2
    QQ               = 2
    model_dir        = './models/magnitude_predictors'
    model_filename   = './models/magnitude_predictors/{random_id}_model_capture_distribution.h5'
    B                = 30
    num_RIS_elements = 21
    rho              : np.ndarray = field(init=False)
    SNR_dB           = 15.
    SNR              : float = field(init=False)
    sigma_sq         : float = field(init=False)


    def __post_init__(self):
        self.SNR      = dB2power(self.SNR_dB)
        self.rho      = 1.0/np.sqrt(self.B) * np.ones((self.B,))
        mag_rho       = np.sqrt(np.sum(np.abs(self.rho)**2))
        self.sigma_sq = mag_rho / self.SNR


    def get_model_filename(self, random_id):
        return self.model_filename.format(random_id=random_id)


def mat2dict(mat):
    L_pert = mat['L_pert'][0, 0]
    PertOrient = mat['PertOrient']
    RIS_configs = mat['RIS_configs']
    Transmission = mat['Transmission']
    freq = mat['freq'].flatten()

    data = dict()
    data['L_pert'] = L_pert
    data['PertOrient'] = PertOrient
    data['RIS_config'] = RIS_configs
    data['Transmission'] = Transmission
    data['Frequency'] = freq

    return data


def train_test_split_by_phases(data, test_size=0.2):
    rng = np.random.default_rng(seed=333333)
    phases_mask = rng.binomial(1, size=data['Transmission'].shape[1], p=1 - test_size)

    RIS_configs_train = data['RIS_config'][:, phases_mask == 1, :]
    Transmissions_train = data['Transmission'][:, phases_mask == 1, :]

    RIS_configs_test = data['RIS_config'][:, phases_mask == 0, :]
    Transmissions_test = data['Transmission'][:, phases_mask == 0, :]

    return RIS_configs_train, Transmissions_train, RIS_configs_test, Transmissions_test


def load_data(setup: Setup, together=False):

    filename = FILENAME_FORMAT.format(setup.QQ, setup.LL)
    mat = scipy.io.loadmat(filename)
    data = mat2dict(mat)

    (RIS_configs_train, Transmissions_train,
     RIS_configs_test, Transmissions_test) = train_test_split_by_phases(data)

    mag_response_squared_train = np.power(np.absolute(Transmissions_train), 2)
    mag_response_squared_test = np.power(np.absolute(Transmissions_test), 2)

    avg_squared_mag_response_train = np.mean(mag_response_squared_train, axis=0)
    avg_squared_mag_response_test = np.mean(mag_response_squared_test, axis=0)

    RIS_configs_train = RIS_configs_train[0, :, :].astype(np.float32)
    RIS_configs_test = RIS_configs_test[0, :, :].astype(np.float32)



    if together:
        RIS_profiles  = np.vstack([RIS_configs_train, RIS_configs_test])
        mag_responses = np.concatenate([avg_squared_mag_response_train, avg_squared_mag_response_test])

        return RIS_profiles, mag_responses

    else:
        return RIS_configs_train, avg_squared_mag_response_train, RIS_configs_test, avg_squared_mag_response_test




def compute_capacity(mag_H_f_sq, rho, sigma_sq):
    c = np.sum(np.log2(1 + (mag_H_f_sq * rho)/(sigma_sq)))
    return c


