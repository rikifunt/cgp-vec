# Example of using cgp-vec for symbolic regression.

from typing import Any, Union, Tuple, List, Dict, Optional, Iterable, Callable

from tqdm import tqdm
import torch
import cgpv

def mse(x, y):
    reduced_dims = tuple(range(2, x.dim()))
    return torch.mean((x - y)**2, dim=reduced_dims)

# the original loss used for Koza regression problems
def koza_regression_loss(x, y):
    reduced_dims = tuple(range(2, x.dim()))
    return torch.sum((x - y).abs(), dim=reduced_dims)

# A single function to perform multiple steps of symbolic regression with a 
# (mu+lambda)-ES. Uses tqdm for the progress bar.
def plus_regression(
        n_steps: int, mutation_rate: float, n_populations: int, n_parents: int,
        n_offspring: int, n_hidden: int, input: torch.Tensor,
        true_output: torch.Tensor,
        primitive_functions: List[cgpv.PrimitiveType],
        primitive_arities: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] \
            = None, # mse
        observe_step: Optional[Callable] = None) -> cgpv.Populations:
    loss = loss or mse
    device = None if generator is None else generator.device
    input = input.to(device)
    true_output = true_output.to(device)
    primitive_arities = primitive_arities.to(device)
    populations = cgpv.Populations.random(
        n_populations=n_populations, pop_size=n_parents, n_inputs=1,
        n_outputs=1, n_hidden=n_hidden, primitive_functions=primitive_functions,
        primitive_arities=primitive_arities, descending_fitness=False,
        generator=generator, device=device)
    populations.fitnesses = loss(populations(input), true_output)
    pbar = tqdm(range(n_steps))
    for i in pbar:
        parents = populations.roulette_wheel(n_rounds=n_offspring,
                                             generator=generator)
        offspring = parents.mutate(rate=mutation_rate, generator=generator)
        offspring.fitnesses = loss(offspring(input), true_output)
        populations = offspring.plus_selection(populations)
        pbar.set_postfix({'loss': f'{populations.fitnesses.mean():0.2f} +- {populations.fitnesses.std():0.2f}'})
        if observe_step is not None:
            observe_step(step=i, observations={
                'dnas': populations.dnas, 'outputs': populations(input), 'losses': populations.fitnesses
              })
    return populations


def main():
    #device = torch.device('cuda')
    device = torch.device('cpu')

    koza_primitives = [
        lambda x: x[0] + x[1],
        lambda x: x[0] * x[1],
        lambda x: x[0] - x[1],
        lambda x: x[0] / x[1]
    ]
    koza_primitive_arities = torch.tensor([2, 2, 2, 2])
    
    # Instantiate 2 Koza problems (Koza-2 and Koza-3). To switch between the 
    # two, pass the appropriate koza*_outputs to the true_output parameter of 
    # plus_regression.
    koza_inputs = torch.linspace(-1., 1., 50).expand(1, -1)
    koza2_target = lambda x: x[0]**5 - 2*x[0]**3 + x[0]
    koza3_target = lambda x: x[0]**6 - 2*x[0]**4 + x[0]**2
    koza2_outputs = koza2_target(koza_inputs)
    koza3_outputs = koza3_target(koza_inputs)

    # big populations like in the Gecco '07 paper on real-valued CGP
    populations = plus_regression(
        n_steps=500, mutation_rate=0.2, n_populations=100, n_parents=48,
        n_offspring=50, n_hidden=10, input=koza_inputs,
        true_output=koza3_outputs, primitive_functions=koza_primitives,
        primitive_arities=koza_primitive_arities,
        generator=cgpv.seeded_generator(seed=42, device=device),
        loss=koza_regression_loss, observe_step=None,)


if __name__ == '__main__':
    main()