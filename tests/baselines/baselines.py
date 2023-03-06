from functools import partial
from typing import Any, Callable, Dict, Tuple
import json

import numpy as np
import cgp

from tests.tar import Tar


def run_classic(
    # Population params
    seed: int,
    n_parents: int,
    # Genome params
    n_inputs: int,
    n_outputs: int,
    n_columns: int,
    n_rows: int,
    levels_back: int,
    primitives: Tuple[str],
    # EA params
    n_offsprings: int,
    tournament_size: int,
    mutation_rate: float,
    n_processes: int,
    # Evolve params
    objective: Callable,
    generations: int,
) -> np.ndarray:
    pop = cgp.Population(
        seed=seed,
        n_parents=n_parents,
        genome_params=dict(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_columns=n_columns,
            n_rows=n_rows,
            levels_back=levels_back,
            primitives=primitives,
        )
    )
    ea = cgp.ea.MuPlusLambda(
        n_offsprings=n_offsprings,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        n_processes=n_processes,
    )

    fitness_history = []

    def recording_callback(pop):
        fitness_history.append(pop.champion.fitness)

    cgp.evolve(
        pop,
        objective=objective,
        ea=ea,
        max_generations=generations,
        print_progress=True,
        callback=recording_callback
    )

    return np.array(fitness_history)


def symbolic_regression_examples(start=-4, end=4, n_examples=1000) -> np.ndarray:
    np.random.seed(1234)
    examples = np.random.uniform(start, end, n_examples)
    return examples


def toy_target(x):
    return x ** 2 + 1.0


def symbolic_regression_neg_mse(individual, examples, f_target):
    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = examples.shape[0]

    f = individual.to_func()
    loss = 0
    for x in examples:
        # the callable returned from `to_func` accepts and returns
        # lists; accordingly we need to pack the argument and unpack
        # the return value
        y = f(x)
        loss += (f_target(x) - y) ** 2

    individual.fitness = -loss / n_function_evaluations

    return individual


def symbolic_regression_baseline(
    baseline_name: str,
    parameters: Dict[str, Any],
    seeds_fpath: str = 'tests/baselines/seeds.txt'
) -> None:

    losses = {
        'symbolic_regression_neg_mse': symbolic_regression_neg_mse,
    }

    target_functions = {
        'toy': toy_target,
    }

    primitives = {
        'add': cgp.Add,
        'sub': cgp.Sub,
        'mul': cgp.Mul,
        'const': cgp.ConstantFloat,
    }

    with open(seeds_fpath, mode='r') as seeds_file:
        seeds = np.array([int(line) for line in seeds_file])

    examples = symbolic_regression_examples()

    mapped_params = parameters.copy()
    mapped_params['objective'] = partial(
        losses[mapped_params.pop('loss')],
        f_target=target_functions[mapped_params.pop('target_function')],
        examples=examples
    )
    mapped_params['primitives'] = tuple(primitives[p] for p in parameters.pop('primitives'))

    fitness_histories = []
    for seed in seeds:
        fitness_histories.append(run_classic(seed=seed, **mapped_params))
    fitness_histories = np.array(fitness_histories)

    def save_json(d, f):
        s = json.dumps(d)
        bs = bytes(s, 'ascii')
        f.write(bs)

    np_save = lambda a, f: np.save(f, a)

    with Tar(filename=f'{baseline_name}.tar', mode='w') as tar:
        tar.add(name='parameters.json', entry=parameters, save_func=save_json)
        tar.add(name='examples.npy', entry=examples, save_func=np_save)
        tar.add(name='seeds.npy', entry=seeds, save_func=np_save)
        tar.add(name='fitness_history.npy', entry=fitness_histories, save_func=np_save)


if __name__ == '__main__':
    parameters = dict(
        # Population params
        n_parents=1,
        # Genome params
        n_inputs=1,
        n_outputs=1,
        n_columns=12,
        n_rows=1,
        levels_back=5,
        primitives=('add', 'sub', 'mul', 'const'),
        # EA params
        n_offsprings=4,
        tournament_size=1,
        mutation_rate=0.03,
        n_processes=1, # > 1 doesn't work for some reason
        # Evolve params
        loss='symbolic_regression_neg_mse',
        target_function='toy',
        generations=100,
    )

    symbolic_regression_baseline(
        'tests/baselines/toy',
        parameters=parameters,
        seeds_fpath='tests/baselines/seeds.txt'
    )
