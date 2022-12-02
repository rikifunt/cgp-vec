from pytest import mark
import torch

import cgpv


class TestPhenotype:
    def test_eval_populations(self):

        target = lambda x: x[0] ** 2 - x[0]
        prims = [
            lambda x: x[0] + x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] - x[1],
            lambda x: x[0] / x[1],
        ]
        arities = torch.tensor([2, 2, 2, 2])

        dnas = torch.tensor(
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
            dtype=torch.long,
        )
        n_hidden = 2

        input = torch.linspace(-5.0, 5.0, steps=50).expand(1, -1)
        true_output = target(input)

        outputs = cgpv.eval_populations(
            input=input,
            dnas=dnas,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_arities=arities,
            max_arity=2,
            primitive_functions=prims,
        )
        assert torch.all(true_output == outputs)

    def test_crazy_dna(self):
        def mse(x, y):
            reduced_dims = tuple(range(2, x.dim()))
            return torch.mean((x - y) ** 2, dim=reduced_dims)

        loss = mse

        koza3_target = lambda x: x[0] ** 6 - 2 * x[0] ** 4 + x[0] ** 2
        primitive_functions = [
            lambda x: x[0] + x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] - x[1],
            lambda x: x[0] / x[1],
        ]
        primitive_arities = torch.tensor([2, 2, 2, 2])
        input = torch.linspace(-1.0, 1.0, 50).expand(1, -1)
        koza3_output = koza3_target(input)

        n_inputs, n_outputs = 1, 1
        n_hidden = 10

        # this gives a inf loss (bot not inf outputs) on koza-3
        dnas = torch.tensor(
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
        outputs = cgpv.eval_populations(
            input=input,
            dnas=dnas,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_arities=primitive_arities,
            max_arity=2,
            primitive_functions=primitive_functions,
        )
        # print(f'max f = {true_output.max()}, min f = {true_output.min()}')
        # print(f'max y = {outputs.max()}, min y = {outputs.min()}')
        assert torch.allclose(simple_true_output, outputs)
        assert torch.all(true_output == outputs)
        # self.assertFalse(torch.any(loss(outputs, koza3_output).isinf()))


class TestPopulations:
    def test_call(self) -> None:
        generator = cgpv.seeded_generator(42)
        prims = [
            lambda x: x[0] + x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] - x[1],
            lambda x: x[0] / x[1],
        ]
        arities = [2, 2, 2, 2]
        populations = cgpv.Populations.random(
            5, 10, 1, 1, 10, prims, arities, generator=generator
        )
        input = torch.linspace(-1.0, 1.0, steps=50).expand(1, -1)
        outputs = populations(input)
        true_outputs = cgpv.eval_populations(
            input=input,
            dnas=populations._dnas,
            n_inputs=1,
            n_outputs=1,
            n_hidden=10,
            primitive_arities=torch.tensor(arities),
            max_arity=2,
            primitive_functions=prims,
        )
        assert torch.all(outputs == true_outputs)

    @mark.slow
    def test_regression_tensor_sanity(self) -> None:

        rng = cgpv.seeded_generator(42)
        device = rng.device

        def mse(x, y):
            reduced_dims = tuple(range(2, x.dim()))
            return torch.mean((x - y) ** 2, dim=reduced_dims)

        loss = mse

        target = lambda x: x[0] ** 6 - 2 * x[0] ** 4 + x[0] ** 2
        primitive_functions = [
            lambda x: x[0] + x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] - x[1],
            lambda x: x[0] / x[1],
        ]
        primitive_arities = torch.tensor([2, 2, 2, 2], device=device)
        input = torch.linspace(-1.0, 1.0, 50, device=device).expand(1, -1)
        true_output = target(input)

        n_populations = 5
        n_parents, n_offspring = 50, 48
        n_hidden = 10
        mutation_rate = 0.2

        populations = cgpv.Populations.random(
            n_populations=n_populations,
            pop_size=n_parents,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_functions=primitive_functions,
            primitive_arities=primitive_arities,
            descending_fitness=False,
            generator=rng,
            device=device,
        )

        assert populations.validate(raise_=False)

        assert not (torch.any(populations(input).isnan()))

        populations.fitnesses = loss(populations(input), true_output)

        assert not (torch.any(populations.fitnesses.isnan()))

        n_steps = 500

        for i in range(n_steps):

            W = populations.fitnesses
            W = 1.0 / W
            assert not (torch.any(W.isnan() | W.isinf())), f"Degenerate weights at step {i}"

            parents = populations.roulette_wheel(n_rounds=n_offspring, generator=rng)

            assert parents.validate(raise_=False)

            offspring = parents.mutate(rate=mutation_rate, generator=rng)

            assert offspring.validate(raise_=False)

            assert not (torch.any(offspring(input).isnan())), f"NaN offspring output at step {i}"

            offspring.fitnesses = loss(offspring(input), true_output)

            assert not (torch.any(offspring.fitnesses.isnan())), f"NaN offspring fitnesses at step {i}"

            populations = offspring.plus_selection(populations)

            assert populations.validate(raise_=False)

    @mark.slow
    def test_koza3_parity(self) -> None:
        seed = 42

        def mse(x, y):
            reduced_dims = tuple(range(2, x.dim()))
            return torch.mean((x - y) ** 2, dim=reduced_dims)

        loss = mse

        target = lambda x: x[0] ** 6 - 2 * x[0] ** 4 + x[0] ** 2
        primitive_functions = [
            lambda x: x[0] + x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] - x[1],
            lambda x: x[0] / x[1],
        ]
        primitive_arities = torch.tensor([2, 2, 2, 2])
        input = torch.linspace(-1.0, 1.0, 50).expand(1, -1)
        true_output = target(input)

        self.assert_regression_parity(
            seed=seed,
            loss=loss,
            primitive_functions=primitive_functions,
            primitive_arities=primitive_arities,
            input=input,
            true_output=true_output,
        )

    @mark.slow
    def test_koza2_parity(self) -> None:
        seed = 42

        def mse(x, y):
            reduced_dims = tuple(range(2, x.dim()))
            return torch.mean((x - y) ** 2, dim=reduced_dims)

        loss = mse

        target = lambda x: x[0] ** 5 - 2 * x[0] ** 3 + x[0]
        primitive_functions = [
            lambda x: x[0] + x[1],
            lambda x: x[0] * x[1],
            lambda x: x[0] - x[1],
            lambda x: x[0] / x[1],
        ]
        primitive_arities = torch.tensor([2, 2, 2, 2])
        input = torch.linspace(-1.0, 1.0, 50).expand(1, -1)
        true_output = target(input)

        self.assert_regression_parity(
            seed=seed,
            loss=loss,
            primitive_functions=primitive_functions,
            primitive_arities=primitive_arities,
            input=input,
            true_output=true_output,
        )

    def assert_regression_parity(
        self, seed, loss, primitive_functions, primitive_arities, input, true_output
    ) -> None:

        rng = cgpv.seeded_generator(seed)
        test_rng = cgpv.seeded_generator(seed)
        device = rng.device

        assert torch.equal(rng.get_state(), test_rng.get_state())

        n_populations = 5
        n_parents, n_offspring = 50, 48
        n_hidden = 10
        mutation_rate = 0.2

        populations = cgpv.Populations.random(
            n_populations=n_populations,
            pop_size=n_parents,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_functions=primitive_functions,
            primitive_arities=primitive_arities,
            descending_fitness=False,
            generator=rng,
            device=device,
        )

        conf = populations.configuration()

        assert not conf.descending_fitness

        populations.fitnesses = loss(populations(input), true_output)

        n_alleles = cgpv.count_alleles(
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            n_primitives=len(primitive_functions),
            max_arity=2,
            device=device,
        )
        assert torch.equal(n_alleles, conf.n_alleles)

        dnas = cgpv.random_populations(
            n_populations=n_populations,
            pop_size=n_parents,
            n_alleles=n_alleles,
            generator=test_rng,
        )
        assert torch.equal(dnas, populations.dnas)

        assert torch.equal(rng.get_state(), test_rng.get_state())

        outputs = cgpv.eval_populations(
            input=input,
            dnas=dnas,
            n_inputs=1,
            n_outputs=1,
            n_hidden=n_hidden,
            primitive_arities=primitive_arities,
            primitive_functions=primitive_functions,
            max_arity=2,
        )
        assert torch.equal(populations(input), outputs)
        losses = loss(outputs, true_output)
        assert torch.equal(losses, populations.fitnesses)

        n_steps = 500

        assert torch.equal(rng.get_state(), test_rng.get_state())

        for i in range(n_steps):

            assert torch.equal(
                rng.get_state(), test_rng.get_state()
            ), f"Wrong RNG state at step {i} [0]"

            parents = populations.roulette_wheel(n_rounds=n_offspring, generator=rng)

            parent_dnas = cgpv.roulette_wheel(
                n_rounds=n_offspring,
                items=dnas,
                weights=1.0 / losses,
                normalize_weights=True,
                generator=test_rng,
            )
            assert torch.equal(parents.dnas, parent_dnas), f"Wrong parents at step {i}"

            assert torch.equal(
                rng.get_state(), test_rng.get_state()
            ), f"Wrong RNG state at step {i} [1]"

            offspring = parents.mutate(rate=mutation_rate, generator=rng)

            offspring_dnas = cgpv.mutate(
                dnas=parent_dnas,
                rate=mutation_rate,
                n_alleles=n_alleles,
                generator=test_rng,
            )

            assert torch.equal(
                rng.get_state(), test_rng.get_state()
            ), f"Wrong RNG state at step {i} [2]"

            assert torch.equal(
                offspring.dnas, offspring_dnas
            ), f"Wrong offspring at step {i}"

            offspring.fitnesses = loss(offspring(input), true_output)

            offspring_outputs = cgpv.eval_populations(
                input=input,
                dnas=offspring_dnas,
                n_inputs=1,
                n_outputs=1,
                n_hidden=n_hidden,
                primitive_arities=primitive_arities,
                primitive_functions=primitive_functions,
                max_arity=2,
            )
            offspring_losses = loss(offspring_outputs, true_output)

            assert torch.equal(
                offspring(input), offspring_outputs
            ), f"Wrong offspring eval at step {i}"
            assert torch.equal(
                offspring.fitnesses, offspring_losses
            ), f"Wrong offspring fitnesses at step {i}"
            assert torch.equal(populations.dnas, dnas)
            assert torch.equal(populations.fitnesses, losses)

            populations = offspring.plus_selection(populations)

            dnas, losses = cgpv.plus_selection(
                parents=dnas,
                parent_fitnesses=losses,
                offspring=offspring_dnas,
                offspring_fitnesses=offspring_losses,
                descending=False,
            )

            assert torch.equal(dnas, populations.dnas), f"Wrong nextgen at step {i}"
            assert torch.equal(
                losses, populations.fitnesses
            ), f"Wrong nextgen fitnesses at step {i}"
