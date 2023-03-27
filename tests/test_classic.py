from pytest import mark
import torch

from cgpv.torch import seeded_generator
import cgpv.classic as classic
import cgpv.evo.selection as selection


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
def test_regression_api_parity() -> None:
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
    minimize = True

    seed: int = 42
    rng: torch.Generator = seeded_generator(seed)
    device: torch.device = rng.device
    test_rng: torch.Generator = seeded_generator(seed, device=device)
    dtype: torch.dtype = torch.long

    examples: torch.Tensor = torch.linspace(-1.0, 1.0, 50, device=device).expand(1, -1)
    true_output: torch.Tensor = target(examples)

    n_populations: int = 5
    pop_size: int = 50
    n_offspring: int = 48
    tournament_size: int = pop_size // 2
    n_inputs: int = 1
    n_outputs: int = 1
    n_hidden: int = 10
    max_arity: int = max(primitives.values())
    mutation_rate: float = 0.2

    def loss_fitness(phenotypes):
        return loss(phenotypes(examples), true_output)

    cgp = classic.Cgp(
        fitness=loss_fitness,
        minimize=True,
        primitives=primitives,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        n_populations=n_populations,
        pop_size=pop_size,
        n_offspring=n_offspring,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        device=device,
        dtype=dtype,
    )

    arities = torch.tensor(tuple(primitives.values()), device=device)

    n_alleles: torch.Tensor = classic.count_alleles(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        n_primitives=len(primitives),
        max_arity=max_arity,
        dtype=torch.long,
    )
    assert torch.all(cgp.n_alleles == n_alleles)

    genomes: torch.Tensor = classic.random_populations(
        n_populations=n_populations,
        pop_size=pop_size,
        n_alleles=n_alleles,
        dtype=dtype,
        generator=test_rng,
    )

    outputs: torch.Tensor = classic.eval_populations(
        input=examples,
        genomes=genomes,
        primitive_arities=arities,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_functions=primitives,
        max_arity=max_arity,
    )

    fitnesses = loss(outputs, true_output)

    cgp.run(generations=1, generator=rng)

    n_steps = 500

    for _ in range(n_steps):

        assert torch.all(cgp.genomes == genomes)
        assert torch.all(cgp.phenotypes(examples) == outputs)
        assert torch.all(cgp.fitnesses == fitnesses)

        parents, _ = selection.tournaments(
            items=genomes,
            scores=fitnesses,
            n_tournaments=n_offspring,
            tournament_size=tournament_size,
            minimize=minimize,
            generator=test_rng,
        )

        offspring = classic.mutate(
            genomes=parents,
            rate=mutation_rate,
            n_alleles=n_alleles,
            generator=test_rng,
        )

        offspring_outputs = classic.eval_populations(
            input=examples,
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
            descending=not minimize,
        )

        cgp.run(generations=1, generator=rng)
        assert torch.all(cgp.parents == parents)
        assert torch.all(cgp.offspring == offspring)
        assert torch.all(cgp.offspring_phenotypes(examples) == offspring_outputs)
        assert torch.all(cgp.offspring_fitnesses == offspring_fitnesses)

    assert torch.all(cgp.genomes == genomes)
    assert torch.all(cgp.phenotypes(examples) == outputs)
    assert torch.all(cgp.fitnesses == fitnesses)
