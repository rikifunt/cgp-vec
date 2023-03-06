import json
import numpy as np

from pytest import mark
import torch

from cgpv.torch import seeded_generator
import cgpv.cgp.classic as classic
import cgpv.evo.selection as selection

from tests.tar import Tar, np_load


class Test_eval_populations:
    def test_simple(self):
        target = lambda x: x[0] ** 2 - x[0]
        primitives = {
            lambda x: x[0] + x[1]: 2,
            lambda x: x[0] * x[1]: 2,
            lambda x: x[0] - x[1]: 2,
            lambda x: x[0] / x[1]: 2,
        }
        primitive_functions = list(primitives.keys())
        primitive_arities = torch.tensor([primitives[f] for f in primitive_functions])
        input = torch.linspace(-5.0, 5.0, steps=50).expand(1, -1)
        true_output = target(input)

        n_hidden = 2
        genomes = torch.tensor(
            [
                [
                    [0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0],
                    [0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0],
                    [0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0],
                ],
                [
                    [0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0],
                    [0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0],
                    [0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0],
                ],
            ],
        )

        outputs = classic.eval_populations(
            genomes=genomes,
            input=input,
            primitive_arities=primitive_arities,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_functions=primitive_functions,
        )
        assert torch.all(true_output == outputs)

    def test_crazy_dna(self):
        primitives = {
            lambda x: x[0] + x[1]: 2,
            lambda x: x[0] * x[1]: 2,
            lambda x: x[0] - x[1]: 2,
            lambda x: x[0] / x[1]: 2,
        }
        primitive_functions = list(primitives.keys())
        primitive_arities = torch.tensor([primitives[f] for f in primitive_functions])
        input = torch.linspace(-1.0, 1.0, 50).expand(1, -1)

        def target(x):
            z1 = x[0] * x[0]
            z2 = z1 * z1
            z3 = z2 * z2
            z5 = z3 * z3
            z6 = z2 / z5
            y0 = z6
            return y0

        simple_target = lambda x: 1 / x[0] ** 12
        simple_true_output = simple_target(input)
        true_output = target(input)

        n_hidden = 10
        # this gives a inf loss (bot not inf outputs) on koza-3
        genomes = torch.tensor(
            [
                [
                    [
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,  # z1 = x0 * x0 = x0^2
                        1,
                        1,
                        1,  # z2 = z1 * z1 = x0^4
                        1,
                        2,
                        2,  # z3 = z2 * z2 = z2^2 = x0^8
                        2,
                        2,
                        1,
                        1,
                        3,
                        3,  # z5 = z3 * z3 = z3^2 = x0^16
                        3,
                        2,
                        5,  # z6 = z2 / z5 = x0^4 / x0^16 = x0^(-12)
                        3,
                        4,
                        2,
                        2,
                        4,
                        0,
                        2,
                        4,
                        8,
                        2,
                        2,
                        2,
                        0,
                        6,
                        0,  # y0 = z6 = x0^(-12)
                    ]
                ]
            ]
        )

        outputs = classic.eval_populations(
            genomes=genomes,
            input=input,
            primitive_arities=primitive_arities,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_functions=primitive_functions,
        )
        assert torch.allclose(simple_true_output, outputs)
        assert torch.all(true_output == outputs)


@mark.slow
def test_regression_tensor_sanity() -> None:
    def mse(x, y):
        reduced_dims = tuple(range(2, x.dim()))
        return torch.mean((x - y) ** 2, dim=reduced_dims)

    loss = mse
    primitives = {
        lambda x: x[0] + x[1]: 2,
        lambda x: x[0] * x[1]: 2,
        lambda x: x[0] - x[1]: 2,
        lambda x: x[0] / x[1]: 2,
    }
    target = lambda x: x[0] ** 6 - 2 * x[0] ** 4 + x[0] ** 2

    rng: torch.Generator = seeded_generator(42)
    device: torch.device = rng.device

    input: torch.Tensor = torch.linspace(-1.0, 1.0, 50, device=device).expand(1, -1)
    true_output: torch.Tensor = target(input)

    n_populations: int = 5
    pop_size: int = 50
    n_offspring: int = 48
    n_inputs: int = 1
    n_outputs: int = 1
    n_hidden: int = 10
    max_arity: int = max(primitives.values())
    mutation_rate: float = 0.2

    arities = torch.tensor(tuple(primitives.values()), device=device)

    n_alleles: torch.Tensor = classic.count_alleles(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        n_primitives=len(primitives),
        max_arity=max_arity,
        dtype=torch.long,
    )

    genomes: torch.Tensor = classic.random_populations(
        n_populations=n_populations,
        pop_size=pop_size,
        n_alleles=n_alleles,
        dtype=torch.long,
        generator=rng,
    )

    outputs: torch.Tensor = classic.eval_populations(
        input=input,
        genomes=genomes,
        primitive_arities=arities,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_functions=primitives,
        max_arity=max_arity,
    )

    assert not (torch.any(outputs.isnan()))

    fitnesses = loss(outputs, true_output)

    assert not (torch.any(fitnesses.isnan()))

    n_steps = 500

    for i in range(n_steps):

        parents, _ = selection.tournaments(
            items=genomes,
            scores=fitnesses,
            n_tournaments=n_offspring,
            tournament_size=pop_size // 2,
            minimize=True,
            generator=rng,
        )

        offspring = classic.mutate(
            genomes=parents,
            rate=mutation_rate,
            n_alleles=n_alleles,
            generator=rng,
        )

        offspring_outputs = classic.eval_populations(
            input=input,
            genomes=offspring,
            primitive_arities=arities,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            primitive_functions=primitives,
            max_arity=max_arity,
        )

        assert not (
            torch.any(offspring_outputs.isnan())
        ), f"NaN offspring output at step {i}"

        offspring_fitnesses = loss(offspring_outputs, true_output)

        assert not (
            torch.any(offspring_fitnesses.isnan())
        ), f"NaN offspring fitnesses at step {i}"

        genomes, fitnesses = selection.plus_selection(
            genomes=genomes,
            fitnesses=fitnesses,
            offspring=offspring,
            offspring_fitnesses=offspring_fitnesses,
            descending=False,
        )


@mark.slow
def test_toy_baseline() -> None:
    def mse(x, y):
        reduced_dims = tuple(range(2, x.dim()))
        return torch.mean((x - y) ** 2, dim=reduced_dims)

    with Tar('tests/baselines/toy.tar') as baseline_tar:
        baseline_parameters = baseline_tar.extract('parameters.json', json.load)
        examples = baseline_tar.extract('examples.npy', np_load)
        baseline_fitness_histories = baseline_tar.extract('fitness_history.npy', np_load)

    primitive_map = {
        'add': (lambda x: x[0] + x[1], 2),
        'sub': (lambda x: x[0] - x[1], 2),
        'mul': (lambda x: x[0] * x[1], 2),
        # TODO how to check this from the TAR?
        'const': (lambda x: torch.ones_like(x, dtype=torch.double), 1), # arity = 0 doesn't work for now
    }

    primitives = dict(primitive_map[p_str] for p_str in baseline_parameters['primitives'])

    # "toy" target
    target = lambda x: x[0] ** 2 + 1.0

    loss = mse
    assert baseline_parameters['objective'] == 'symbolic_regression_neg_mse_toy'
    baseline_fitness_histories = -baseline_fitness_histories

    rng: torch.Generator = seeded_generator(42)
    device: torch.device = rng.device

    baseline_fitness_histories = torch.tensor(baseline_fitness_histories, device=device)

    input: torch.Tensor = torch.tensor(examples, device=device).expand(1, -1)
    true_output: torch.Tensor = target(input)

    n_populations: int = baseline_fitness_histories.size(0)
    pop_size: int = baseline_parameters['n_parents']
    tournament_size: int = baseline_parameters['tournament_size']
    n_offspring: int = baseline_parameters['n_offsprings']
    n_inputs: int = baseline_parameters['n_inputs']
    n_outputs: int = baseline_parameters['n_outputs']
    n_hidden: int = baseline_parameters['n_columns']
    assert baseline_parameters['n_rows'] == 1
    max_arity: int = max(primitives.values())
    mutation_rate: float = baseline_parameters['mutation_rate']
    # NOTE: assumed to be number of generations including the first one,
    # so the number of evolutionary steps is actually n_generations - 1.
    n_generations: int = baseline_fitness_histories.size(1)

    arities = torch.tensor(tuple(primitives.values()), device=device)

    fitness_histories = torch.full(
        [n_populations, n_generations],
        float('nan'),
        dtype=baseline_fitness_histories.dtype,
        device=device
    )

    n_alleles: torch.Tensor = classic.count_alleles(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        n_primitives=len(primitives),
        max_arity=max_arity,
        dtype=torch.long,
    )

    genomes: torch.Tensor = classic.random_populations(
        n_populations=n_populations,
        pop_size=pop_size,
        n_alleles=n_alleles,
        dtype=torch.long,
        generator=rng,
    )

    outputs: torch.Tensor = classic.eval_populations(
        input=input,
        genomes=genomes,
        primitive_arities=arities,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_functions=primitives,
        max_arity=max_arity,
    )

    fitnesses = loss(outputs, true_output) # (n_populations, pop_size)

    for i in range(n_generations - 1):
        fitness_histories[:,i] = torch.max(fitnesses, dim=1)[0] # (n_populations,)

        parents, _ = selection.tournaments(
            items=genomes,
            scores=fitnesses,
            n_tournaments=n_offspring,
            tournament_size=tournament_size,
            minimize=True,
            generator=rng,
        )

        offspring = classic.mutate(
            genomes=parents,
            rate=mutation_rate,
            n_alleles=n_alleles,
            generator=rng,
        )

        offspring_outputs = classic.eval_populations(
            input=input,
            genomes=offspring,
            primitive_arities=arities,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            primitive_functions=primitives,
            max_arity=max_arity,
        )

        offspring_fitnesses = loss(offspring_outputs, true_output)

        genomes, fitnesses = selection.plus_selection(
            genomes=genomes,
            fitnesses=fitnesses,
            offspring=offspring,
            offspring_fitnesses=offspring_fitnesses,
            descending=False,
        )

    fitness_histories[:,n_generations-1] = torch.max(fitnesses, dim=1)[0] # (n_populations,)

    np.save('fitness_histories.npy', fitness_histories.numpy())

    baseline_avgs = torch.mean(baseline_fitness_histories, dim=0)
    avgs = torch.mean(fitness_histories, dim=0)

    # TODO use hypothesis checking theory for this
    # TODO this fails anyway, hal-cgp seems to perform a little better,
    #      need to look into it (performance is similar anyway)
    assert torch.all((baseline_avgs - avgs)[2:].abs() < 10)
