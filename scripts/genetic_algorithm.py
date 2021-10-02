from collections import Callable
from dataclasses import field, dataclass
from enum import Enum, auto
from typing import Union, List

import numpy as np
from tqdm import tqdm

def crossover(pop, r_cross):
    N              = pop.shape[0]
    M              = pop.shape[1]

    parents1       = pop[:N // 2, :]
    parents2       = pop[N // 2:, :]
    mask1          = np.random.randint(0, 1, size=(N // 2, M))
    probs_cross    = np.random.uniform(0, 1, size=N // 2)
    cross_allowed  = (probs_cross < r_cross).astype(np.byte)
    cross_allowed2 = np.repeat(cross_allowed, M).reshape(len(cross_allowed), M)
    mask2          = cross_allowed2 * mask1
    mask3          = mask2.astype(np.byte)
    p1_allowed     = np.logical_xor(parents1, mask3)
    p1_not_allowed = np.logical_xor(parents1, 1 - mask3)
    p2_allowed     = np.logical_xor(parents2, mask3)
    p2_not_allowed = np.logical_xor(parents2, 1 - mask3)
    children1      = np.logical_xor(p1_allowed, p2_not_allowed)
    children2      = np.logical_xor(p1_not_allowed, p2_allowed)
    children       = np.vstack((children1, children2))

    return children

def mutate(pop, r_mut):
    probs_mut      = np.random.uniform(0, 1, pop.shape)
    mask4          = (probs_mut < r_mut).astype(np.byte)
    mutated        = (np.logical_xor(pop, mask4)).astype(np.byte)

    return mutated


def create_next_generation(selected, r_cross, r_mut):

    pop = crossover(selected, r_cross)
    pop = mutate(pop, r_mut)

    return pop



def evaluate_population(objective, pop):
    scores = np.zeros(pop.shape[0], dtype=np.float64)
    for i in range(pop.shape[0]):
        scores[i] = objective(pop[i, :])

    return scores



def tournament_selection(scores, k):

    res = np.empty(len(scores), dtype=np.int64)
    for i in range(scores.shape[0]):
        participant_indices = np.random.randint(0, len(scores), k)
        participant_scores  = scores[participant_indices]
        winner_i            = np.argmax(participant_scores)
        winner_actual_index = participant_indices[winner_i]
        res[i]              = winner_actual_index
    return res


def run_single_generation_step(objective, prev_pop, prev_scores, k, r_cross, r_mut):

    new_population_idx  = tournament_selection(prev_scores, k)
    selected            = prev_pop[new_population_idx, :]
    next_pop            = create_next_generation(selected, r_cross, r_mut)
    next_scores         = evaluate_population(objective, next_pop)

    gen_best_index      = np.argmax(next_scores)
    gen_best_individual = next_pop[gen_best_index]
    gen_best_score      = next_scores[gen_best_index]

    return gen_best_individual, gen_best_score, next_pop, next_scores






class GAHistory:

    class HistoryType(Enum):
        FULL           = "full"
        BEST_ONLY      = "best_only"

    class StoppingReason(Enum):
        MAX_ITERS      = auto
        NO_IMPROVEMENT = auto
        USER_INTERRUPT = auto
        EXCEPTION      = auto


    _type                       : HistoryType
    stopping_reason             : Union[StoppingReason, None] = None
    iterations_run              : int                         = 0

    optimal_value               : np.float32                  = None
    optimal_solution            : np.ndarray                  = None

    total_evaluations           : int                         = 0
    iterations_with_improvement : int                         = 0
    last_obj_value_tol          : float                       = 0.
    total_mutations             : int                         = 0
    total_crossovers            : int                         = 0

    best_generation_scores      : List[np.float32]            = field(default_factory=list)
    best_generation_individuals : List[np.ndarray]            = field(default_factory=list)

    all_generation_scores       : List[np.ndarray]            = field(default_factory=list)
    all_generation_populations  : List[np.ndarray]            = field(default_factory=list)



@dataclass()
class GAParams:

    # parameters
    objective_function        : Callable
    input_length              : float                               = None
    r_cross                   : float                               = .9
    r_mut                     : float                               = .1

    # initialization
    initial_population        : [np.ndarray]                        = None


    # stopping
    max_iters                 : int                                 = 1000
    iters_without_improvement : int                                 = 3
    obj_value_tol_normalized  : float                               = 10e-5


    # resuming run
    previous_run_history      : GAHistory                           = None


    # control
    verbose                   : int                                 = 0
    record_history            : Union[str]                          = False
    tqdm                      : bool                                = False
    seed                      : int                                 = None
    numba_parallel            : bool                                = False
    numba_fastmath            : bool                                = True


    def __post_init__(self):

        if not ( 0<= self.r_cross <= 1):
            raise ValueError("Probability of crossover must be in [0,1].")

        if not ( 0<= self.r_mut <= 1 ):
            raise ValueError("Probability of mutation must be in [0,1].")

        if self.input_length is not None and self.input_length <= 0:
            raise ValueError("Non positive value passed as input length.")

        if self.max_iters <= 0:
            raise ValueError("Maximum number of iterations must be at least 1.")

        if self.iters_without_improvement <= 0:
            raise ValueError("Number of iterations without improvement before stopping must be at least 1 or None.")

        if not (0<= self.obj_value_tol_normalized <= 1.) :
            raise ValueError("Normalized tolerance for comparing objective function values must be in (0,1) or None.")

        if self.verbose > 2:
            raise ValueError(f"Unknown verbosity level {self.verbose}.")

        if self.record_history is not None and self.record_history is not False:
            if self.record_history.lower() not in GAHistory.HistoryType.__members__.values():
                raise ValueError('Values for `record_history` must be either False or one of {}'.format(
                    GAHistory.HistoryType.__members__.values()
                ))

        if self.seed < 0:
            raise ValueError(f'Seed cannot be negative (passed {self.seed}).')

        if self.initial_population is not None:

            self.initial_population = np.array(self.initial_population, dtype=np.byte)

            if self.initial_population.ndim != 2:
                raise ValueError(f"Initial population array must have 2 dimensions (found {self.initial_population.ndim}).")

            if self.input_length is None and self.initial_population.shape[1] != self.input_length:
                raise ValueError("Value of `input_length` and number of columns of `initial_population` array do not agree "+\
                                 f"(Passed {self.input_length} and {self.initial_population.shape[1]}).")

        else:
            if self.input_length is None:
                raise ValueError("At least one of `input_length`, `initial_population` arguments is required.")









def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, k=3, verbose=0):
    np.random.seed(12)

    pop0                    = np.random.randint(0, 2, size=(n_pop, n_bits))
    pop                     = pop0.astype(np.byte)
    scores                  = evaluate_population(objective, pop)
    best_individual         = pop[0, :]
    best_score              = scores[0]

    for gen_number in range(n_iter):

        (gen_best_individual,
        gen_best_score,
        pop,
        scores)             = run_single_generation_step(objective, pop, scores, k, r_cross, r_mut)


        if gen_best_score > best_score:
            best_individual, best_score = gen_best_individual, gen_best_score
            if verbose>=1: print('Generation ',gen_number,': Updated best ga_capacity to: ',best_score,)
        else:
            if verbose>=2: print('Generation ',gen_number,': Best ga_capacity: ',best_score)


    return best_individual, best_score







if __name__ == '__main__':

    def main():

        np.random.seed(42)
        N = 100
        H = np.random.randn(N)
        G = np.random.randn(N)
        F = np.random.randint(0,2,N)-1
        h = np.random.randn(1)





        def snr(F):
            F = F.astype(np.float64)
            return (np.dot(H*G, F) + h)[0]

        global PARALLEL

        PARALLEL = True
        for _ in tqdm(range(100)):
            sol, score = genetic_algorithm(snr, N, 500, 100, 0.9, 0.01)

        # print(ga_ris_profile)
        # print(ga_capacity)



        PARALLEL = False
        for _ in tqdm(range(100)):
            sol, score = genetic_algorithm(snr, N, 500, 100, 0.9, 0.01)

        # print(ga_ris_profile)
        # print(ga_capacity)


    main()