from scripts.common import Setup, load_data, compute_capacity

import numpy as np
import itertools
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



print(sns.__version__)

def display_facet_grid():

    labels = np.chararray(9, itemsize=len("LL: 1, QQ: 1"))
    capacities = np.empty((9, 500))

    i = 0
    for ll in [1,2,3]:
        for qq in [1,2,3]:

            labels[i] = f"LL: {ll}, QQ: {qq}"

            setup = Setup(ll,qq,20)
            RIS_profiles, mag_responses = load_data(setup, together=True)
            for j in range(mag_responses.shape[0]):
                capacities[i, j] = compute_capacity(mag_responses[j,:], setup.rho, setup.sigma_sq)


            print(f"rates for {labels[i]}: [{capacities[i].min()}, {capacities[i].max()}]")
            i += 1


    fig, ax = plt.subplots(3,3, sharey='all', sharex='col')


    text_coords = [[(10,50), (6, 50), (1, 50)],
                   [(11,50), (6, 50), (1, 50)],
                   [(11,50), (6.5, 50), (1.25, 50)]]

    i = 0
    for ll in [1, 2, 3]:
        for qq in [1, 2, 3]:

            sns.histplot(capacities[i], ax=ax[ll-1, qq-1])
            ax[ll - 1, qq - 1].set_title(labels[i].decode('utf-8'))
            ax[ll - 1, qq - 1].set_xlabel('Rate (bps)')

            text = f'''
                    min : {capacities[i].min():.2f}
                    max : {capacities[i].max():.2f}
                    mean: {capacities[i].mean():.2f}
                    std : {capacities[i].std():.2f}
                    range: {capacities[i].max() - capacities[i].min():.2f}
            '''.strip().replace('  ','')

            #ax[ll - 1, qq - 1].text(text_coords[ll-1][qq-1][0], text_coords[ll-1][qq-1][0], text)

            print(f"{labels[i]}\n{text}")






            # sns.boxplot(capacities[i], ax=ax2[ll-1, qq-1])
            # ax2[ll - 1, qq - 1].set_title(labels[i].decode('utf-8'))
            # ax2[ll - 1, qq - 1].set_xlabel('Rate (bps)')


            i += 1


    plt.show()

display_facet_grid()

