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


setup          = Setup(1, 3, 12.5)
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




def neuralnet_statistics(model, num_ris_elements, rho, sigma_sq, samples=10000, plot=False):

    RIS_profiles = np.random.binomial(1, .5, size=(samples, num_ris_elements)).astype(np.float32)
    magnitudes   = np.empty((samples, len(rho)))
    capacities   = np.empty(samples)


    for i in tqdm(range(samples)):
        mag_H_f_sq      = model.predict(RIS_profiles[i,:].reshape((1,-1)))
        capacity        = compute_capacity(mag_H_f_sq, rho, sigma_sq)
        magnitudes[i,:] = mag_H_f_sq
        capacities[i]   = capacity


    if plot:
        plt.figure()
        sns.displot(capacities)
        plt.xlim([23, 24])
        plt.xlabel('$C(\Phi)$')
        plt.show()


        plt.figure()
        sns.boxplot(capacities)
        plt.xlim([23, 24])
        plt.xlabel('$C(\Phi)$')
        plt.show()

    return capacities.mean(), capacities.std()


def compute_capacity_gradient(mag_H_f_sq,
                              rho,
                              sigma_sq,
                              neuralnet_gradients):

    scale = np.sum( rho / (sigma_sq + mag_H_f_sq * rho) )
    #scale = np.sum(rho / (sigma_sq * np.log(2)))
    grad = scale * neuralnet_gradients.flatten()
    return grad


def continuous2binary(x):
    #if x.max() > 1 or x.min() < 0: raise ValueError
    x = x.clip(0, 1)
    return (x>0.5).astype(x.dtype)

def perform_gradient_ascent(model,
                            ris_profile,
                            rho,
                            sigma_sq,
                            epochs,
                            k,
                            learning_rate=10e-2,
                            gradient_clipping=None,
                            reg_lambda=1e-3,
                            verbose=0):

    gradient_wrt_input = K.function(inputs=model.input, outputs=K.gradients(model.output, model.input))

    phi = ris_profile.astype(np.float32)
    initial_capacity = None

    iterator = tqdm(range(epochs)) if verbose >= 1 else range(epochs)

    max_capacity = 0.
    best_phi     = None


    history = {
        'length'        : -1,
        'capacity'      : [],
        'delta_capacity': [],
        'delta_phi'     : [],
        'nn_grad_mag'   : [],
        'grad_mag'      : [],
        'new_best_epoch': [],
        'bit_change_epoch': [],
    }



    for epoch in iterator:
        mag_H_f_sq = model.predict(phi.reshape((1, -1)))
        capacity   = compute_capacity(mag_H_f_sq, rho, sigma_sq)
        nn_grads   = gradient_wrt_input(inputs=phi.reshape((1, -1)))[0]
        nn_grads   = nn_grads[0, :]
        norm_grads = 2 * np.abs(phi)
        grad       = compute_capacity_gradient(mag_H_f_sq, rho, sigma_sq, nn_grads)
        grad       = grad + reg_lambda * norm_grads


        if gradient_clipping is not None: grad = np.clip(grad, -gradient_clipping/len(grad), +gradient_clipping/len(grad))

        phi_next   = phi + learning_rate * grad
        #phi_next   = np.clip(phi_next, 0, 1)




        if capacity > max_capacity:
            max_capacity = capacity
            best_phi     = phi
            history['new_best_epoch'].append(epoch)


        if initial_capacity is None: initial_capacity = capacity



        nn_grad_mag    = np.linalg.norm(nn_grads)
        grad_mag       = np.linalg.norm(grad)
        delta_capacity = capacity - initial_capacity
        delta_phi      = np.linalg.norm(phi_next - phi, 1)


        history['length'] = epoch
        history['nn_grad_mag'].append(nn_grad_mag)
        history['grad_mag'].append(grad_mag)
        history['capacity'].append(capacity)
        history['delta_capacity'].append(delta_capacity)
        history['delta_phi'].append(delta_phi)



        if hamming_distance(continuous2binary(phi), continuous2binary(phi_next)) != 0:
            history['bit_change_epoch'].append(epoch)


        if verbose >= 2:
            tqdm.write(f"[{epoch}/{epochs}] δC = {delta_capacity} | nn grad = {nn_grad_mag} | grad = {grad_mag} | δ phi = {delta_phi}")


        if np.isnan(capacity) or np.isinf(capacity):
            print("[Gradient Descent] WARNING: NaN of Inf value encountered in capacity. Returning last float value.")
            break

        if (epoch+1) % k == 0: phi_next = continuous2binary(phi_next)

        phi = phi_next



    phi             = continuous2binary(best_phi)
    capacity        = compute_capacity(model.predict(phi.reshape((1, -1))), rho, sigma_sq)
    cap_improvement = (capacity / initial_capacity - 1)*100

    if verbose >= 1:
        print(f"\nGradient ascent: Improvement in capacity: {cap_improvement:.3f}%. bits changed: {hamming_distance(phi, ris_profile)}.")

    return phi, capacity, history




def plot_gradient_ascent_history(history):


    x = np.arange(history['length']+1)


    fig, ax = plt.subplots()
    ax.plot(x, history['capacity'])
    ax.vlines(history['bit_change_epoch'], min(history['capacity']), max(history['capacity']))
    #ax.scatter(history['new_best_epoch'], np.array(history['capacity'])[history['new_best_epoch']])
    ax.set_ylabel('Capacity')
    ax.set_xlabel('Iteration')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, history['delta_capacity'])
    ax.vlines(history['bit_change_epoch'], min(history['delta_capacity']), max(history['delta_capacity']))
    ax.set_ylabel('δ capacity')
    ax11 = ax.twinx()
    ax11.plot(x, history['delta_phi'], c='r')
    ax11.set_ylabel('δ Φ',c='r')
    ax.set_xlabel('Iteration')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, history['grad_mag'])
    ax.vlines(history['bit_change_epoch'], min(history['grad_mag']), max(history['grad_mag']))
    ax.set_ylabel('Gradient Magnitude')
    ax22 = ax.twinx()
    ax22.plot(x, history['nn_grad_mag'], c='r')
    ax22.set_ylabel('Neuralnet Gradient Magnitude', c='r')
    ax.set_xlabel('Iteration')
    plt.show()


# # ---------------------------------------------------------------
# #        Statistics
# # --------------------------------------------------------------
known_RIS_profiles, known_avg_mag_responses = load_data(setup, together=True)
known_capacities                            = np.empty(known_avg_mag_responses.shape[0])
for i in range(known_avg_mag_responses.shape[0]):
    known_capacities[i] = compute_capacity(known_avg_mag_responses[i,:], setup.rho, setup.sigma_sq)
gt_mean = known_capacities.mean()
gt_std  = known_capacities.std()
#
# plt.hist(known_capacities)
# plt.show()


# nn_mean, nn_std = neuralnet_statistics(model, setup.num_RIS_elements, setup.rho, setup.sigma_sq, samples=10000)
# print()
# print(f"Capacity statistics of index data     : mean {gt_mean}, std: {gt_std}.")
# print(f"Capacity statistics of trained network: mean {nn_mean}, std: {nn_std}.")
# print()




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




#
# Gradient Descent
#
# best_phi_gd, best_capacity_gd, gd_history  = perform_gradient_ascent(model,
#                                                   ris_profile=rand_best_profile,
#                                                   rho=setup.rho,
#                                                   sigma_sq=setup.sigma_sq,
#                                                   epochs=10000,
#                                                   k=np.infty,
#                                                   learning_rate=1.1e-4,
#                                                   #gradient_clipping = 100,
#                                                   reg_lambda=0,
#                                                   verbose=1)
#
#
# plot_gradient_ascent_history(gd_history)

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



k_best, p_mut_best, pop_best = grid_search_genetic(1000)
# k_best = 6#np.random.choice([3, 5, 10, 20])
# p_mut_best = 0.2#np.random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99])
# pop_best = 200#np.random.choice([50,100,200])
ga_ris_profile, ga_capacity = genetic_algorithm(objective_func, setup.num_RIS_elements, 20000//pop_best, pop_best, .1, p_mut_best, k=k_best)










#
# Indexed data (ground truth capacities)
#
#best_known_capacity  = float(known_capacities.max())



# ------------------------------------- Printing results -------------------------------------------------


print("\nExhaustive search\n-------------------")
print(f'Best RIS profile at SNR {setup.SNR_dB} (dB): {best_profile_exhaustive}.')
print(f' > Capacity: {best_capacity_exhaustive}')
print(f' > {best_capacity_exhaustive/best_capacity_exhaustive} of optimal.')
print(f' > {best_capacity_exhaustive/avg_capacity} of average. (Average is {avg_capacity/best_capacity_exhaustive} of optimal.)')

# print(f'\nIndexed data\n-------------------')
# print(f'> Capacity {best_known_capacity} (NOTE: Computed via known magnitudes and not through the network')
# print(f" > {best_known_capacity / best_capacity_exhaustive} of optimal. ")
# print(f" > {best_known_capacity / avg_capacity} of average. ")


print("\nBest out of Random\n--------------------")
print(f" > {hamming_distance(rand_best_profile, best_profile_exhaustive.astype(int))} bits different from optimal RIS profile.")
print(f" > Capacity: {rand_best_capacity}")
print(f" > {rand_best_capacity/best_capacity_exhaustive} of optimal.")
print(f" > {rand_best_capacity/avg_capacity} of average.")

# print(f'\nGradient Descent\n-------------------')
# print(f" > {hamming_distance(best_phi_gd, best_profile_exhaustive)} bits different from optimal RIS profile.")
# print(f' > Capacity: {best_capacity_gd}')
# print(f" > {best_capacity_gd/best_capacity_exhaustive} of global optimal. ")
# print(f" > {best_capacity_gd/avg_capacity} of average. ")

print(f'\nGenetic Algorithm\n------------------')
print(f" > {hamming_distance(ga_ris_profile, best_profile_exhaustive.astype(int))} bits different from optimal RIS profile.")
print(f' > Capacity: {ga_capacity}')
print(f" > {ga_capacity / best_capacity_exhaustive} of optimal. ")
print(f" > {ga_capacity / avg_capacity} of average. ")