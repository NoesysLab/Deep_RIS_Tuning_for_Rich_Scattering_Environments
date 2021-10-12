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






def random_walk_optimize_capacity(model, rho, sigma_sq, num_evaluations):

    random_profiles = np.random.binomial(1, .5, size=(num_evaluations, setup.num_RIS_elements))
    responses       = model.predict(random_profiles)

    best_capacity = -np.infty
    best_profile  = np.zeros(setup.num_RIS_elements)
    avg_capacity  = 0.

    for i, mag_H_f_sq in enumerate(responses):
        capacity = compute_capacity(mag_H_f_sq, rho, sigma_sq)
        if capacity > best_capacity:
            best_capacity = capacity
            best_profile  = random_profiles[i,:]

        avg_capacity += capacity / num_evaluations

    return best_profile, best_capacity, avg_capacity


def hamming_distance(a, b):
    return np.sum(np.abs(a-b)).astype(int)



def exhaustive_search_maximize_capacity(model, rho, sigma_sq, verbose=0, save_filename=None):

    ris_elements      = setup.num_RIS_elements
    binary_enumerator = itertools.product([0, 1], repeat=ris_elements)

    best_capacity     = -np.infty
    avg_capacity      = 0.0
    best_profile      = None

    if save_filename is not None:
        all_magnitudes = np.empty((2**setup.num_RIS_elements, setup.B), dtype=np.float32)
        all_capacities = np.empty((2**setup.num_RIS_elements,))

    i = 0
    for profile in tqdm(binary_enumerator, total=int(2**ris_elements)):
        mag_H_f_sq = model.predict(np.array(profile).reshape((1,-1)))
        capacity   = compute_capacity(mag_H_f_sq, rho, sigma_sq)

        if verbose >= 1:
            tqdm.write(f"Profile: {profile} | Capacity: {capacity}")

        if save_filename is not None:
            all_magnitudes[i,:] = mag_H_f_sq
            all_capacities[i]   = capacity

        if capacity > best_capacity:
            best_capacity = capacity
            best_profile  = profile

            if verbose >= 1:
                tqdm.write("[New Best!]")
                sleep(1)
        else:
            pass

        i += 1

        avg_capacity += capacity / 2**ris_elements


    if save_filename is not None:
        np.save(all_magnitudes, open(f"{save_filename}_avg_magnitudes.npy", 'w'))
        np.save(all_capacities, open(f"{save_filename}_capacities.npy", 'w'))

    return best_profile, best_capacity, avg_capacity






#
# Genetic Algorithm
#
def objective_func(ris_profile):
    mag_H_f_sq = model.predict(ris_profile.reshape((1,-1)))
    return compute_capacity(mag_H_f_sq, setup.rho, setup.sigma_sq)



def grid_search_genetic(num_evaluations=5000):
    p_mut_vals   = [0.01, 0.02, 0.05, 0.1, 0.2]
    k_vals       = [3, 5, 10, 20]
    pop_sizes    = [50, 100, 200]

    num_variants = len(k_vals) * len(p_mut_vals) * len(pop_sizes)

    val_best    = -np.infty
    params_best = None

    pbar = tqdm(total=num_variants)
    for k in k_vals:
        for p_mut in p_mut_vals:
            for pop in pop_sizes:
                pbar.update()

                _, val = genetic_algorithm(objective_func, setup.num_RIS_elements, num_evaluations // pop, pop, .5, p_mut, k=k)

                tqdm.write(f"Tried k: {k:2d}, p_mut: {p_mut:.2f}, pop: {pop:3d} | Rate: {val:6.4f}")

                if val > val_best:
                    val_best, params_best = val, (k, p_mut, pop)
                    tqdm.write("[updated best]")

    return params_best








#
# Genetic Algorithm
#
k_best, p_mut_best, pop_best = grid_search_genetic(500)
# k_best = 6#np.random.choice([3, 5, 10, 20])
# p_mut_best = 0.2#np.random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99])
# pop_best = 200#np.random.choice([50,100,200])
ga_ris_profile, ga_capacity = genetic_algorithm(objective_func, setup.num_RIS_elements, 20000//pop_best, pop_best, .1, p_mut_best, k=k_best)


#
# Random Search
#
rand_best_profile, rand_best_capacity, avg_capacity = random_walk_optimize_capacity(model, setup.rho, setup.sigma_sq, 5000)


#
# Exhaustive Search
#
#best_profile_exhaustive, _, avg_capacity = exhaustive_search_maximize_capacity(model, setup.rho, setup.sigma_sq, verbose=0)
best_profile_exhaustive = np.array([0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1.])
best_profile_exhaustive = np.array(best_profile_exhaustive, dtype=float)
best_capacity_exhaustive = compute_capacity(model.predict(best_profile_exhaustive.reshape((1,-1)))[0,:], setup.rho, setup.sigma_sq)




# ------------------------------------- Printing results -------------------------------------------------


print("\nExhaustive search\n-------------------")
print(f'Best RIS profile at SNR {setup.SNR_dB} (dB): {best_profile_exhaustive}.')
print(f' > Capacity: {best_capacity_exhaustive}')
print(f' > {best_capacity_exhaustive/best_capacity_exhaustive} of optimal.')
print(f' > {best_capacity_exhaustive/avg_capacity} of average. (Average is {avg_capacity/best_capacity_exhaustive} of optimal.)')


print("\nBest out of Random\n--------------------")
print(f" > {hamming_distance(rand_best_profile, best_profile_exhaustive.astype(int))} bits different from optimal RIS profile.")
print(f" > Capacity: {rand_best_capacity}")
print(f" > {rand_best_capacity/best_capacity_exhaustive} of optimal.")
print(f" > {rand_best_capacity/avg_capacity} of average.")

print(f'\nGenetic Algorithm\n------------------')
print(f" > {hamming_distance(ga_ris_profile, best_profile_exhaustive.astype(int))} bits different from optimal RIS profile.")
print(f' > Capacity: {ga_capacity}')
print(f" > {ga_capacity / best_capacity_exhaustive} of optimal. ")
print(f" > {ga_capacity / avg_capacity} of average. ")