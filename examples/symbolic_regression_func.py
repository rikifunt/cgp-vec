# Example of using cgp-vec for symbolic regression, without using the 
# Populations class.

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
        n_steps: int,
        mutation_rate: float,
        n_populations: int,
        n_parents: int,
        n_offspring: int,
        n_hidden: int,
        input: torch.Tensor,
        true_output: torch.Tensor,
        primitive_functions: List[Callable[[torch.Tensor], torch.Tensor]],
        primitive_arities: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] \
            = None, # mse
        observe_step: Optional[Callable] = None,
        gene_dtype: Optional[torch.dtype] = None, # long
      ):

    loss = loss or mse
    gene_dtype = gene_dtype or torch.long

    # Use the device of the generator if given.
    if generator is not None:
        device = generator.device
        input = input.to(device)
        true_output = true_output.to(device)
        primitive_arities = primitive_arities.to(device)
    else:
        device = input.device

    # Compute preliminary information.
    n_inputs, n_outputs = 1, 1
    n_primitives = len(primitive_functions)
    max_arity = int(torch.max(primitive_arities).item())
    n_alleles = cgpv.count_alleles(
        n_inputs,
        n_outputs,
        n_hidden,
        n_primitives,
        max_arity,
        dtype=gene_dtype,
        device=device
      )
  
    # Generate and evaluate the initial populations.
    dnas = cgpv.random_populations(
        n_populations=n_populations,
        pop_size=n_parents,
        n_alleles=n_alleles,
        generator=generator,
        dtype=gene_dtype
      )
    outputs = cgpv.eval_populations(
        input=input,
        dnas=dnas,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_arities=primitive_arities,
        primitive_functions=primitive_functions,
        max_arity=max_arity
      )
    losses = loss(outputs, true_output)

    pbar = tqdm(range(n_steps))
    for i in pbar:

        # Select parents for reproduction proportionally to their fitnesses.
        parents = cgpv.roulette_wheel(
            n_rounds=n_offspring,
            items=dnas,
            weights=1./losses,
            normalize_weights=True,
            generator=generator,
          )

        # Clone and mutate parents, and evaluate them.
        offspring = cgpv.mutate(
            dnas=parents,
            rate=mutation_rate,
            n_alleles=n_alleles,
            generator=generator,
          )
        offspring_outputs = cgpv.eval_populations(
            input=input,
            dnas=offspring,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            primitive_arities=primitive_arities,
            primitive_functions=primitive_functions,
            max_arity=max_arity
          )
        offspring_losses = loss(offspring_outputs, true_output)

        # Select the parents for the next generation with plus-selection.
        dnas, losses = cgpv.plus_selection(
            parent_fitnesses=losses,
            offspring_fitnesses=offspring_losses,
            parents=dnas,
            offspring=offspring,
            descending=False
        )

        pbar.set_postfix({'loss': f'{losses.mean():0.2f} +- {losses.std():0.2f}'})

        if observe_step is not None:
            observe_step(step=i, observations={
                'dnas': dnas, 'outputs': outputs, 'losses': losses
              })

    return dnas, outputs


def main():
    #device = torch.device('cuda')
    device = torch.device('cpu')

    # Instantiate 2 Koza problems (Koza-2 and Koza-3). To switch between the 
    # two, pass the appropriate koza*_outputs to the true_output parameter of 
    # plus_regression.
    
    koza_primitives = [
        lambda x: x[0] + x[1],
        lambda x: x[0] * x[1],
        lambda x: x[0] - x[1],
        lambda x: x[0] / x[1]
    ]
    koza_primitive_arities = torch.tensor([2, 2, 2, 2])
    
    koza_inputs = torch.linspace(-1., 1., 50).expand(1, -1)
    koza2_target = lambda x: x[0]**5 - 2*x[0]**3 + x[0]
    koza3_target = lambda x: x[0]**6 - 2*x[0]**4 + x[0]**2
    koza2_outputs = koza2_target(koza_inputs)
    koza3_outputs = koza3_target(koza_inputs)

    dnas, outputs = plus_regression(
        n_steps=500,
        mutation_rate=0.2,
        n_populations=100,
        n_parents=50,
        n_offspring=48,
        n_hidden=10,
        input=koza_inputs,
        true_output=koza3_outputs,
        primitive_functions=koza_primitives,
        primitive_arities=koza_primitive_arities,
        generator=cgpv.seeded_generator(seed=42, device=device),
        loss=koza_regression_loss,
        observe_step=None,
        gene_dtype=torch.long,
      )


if __name__ == '__main__':
    main()