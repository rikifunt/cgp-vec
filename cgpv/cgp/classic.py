from typing import Optional, Union, Iterable, Callable

import torch

from cgpv.torch import aranges, randints, random_mask_like


Primitive = Callable[[torch.Tensor], torch.Tensor]


# TODO solve the problem of common arithmetic giving NaN results, e.g.
# division by 0, large multiplication, ...
# The problem is giving users the possibility of using torch operators
# out-of-the-box, without hiding errors that may lead to NaN values.
# - we may keep a list of operators that are known to cause NaNs, but this
#   won't catch lambdas
# - we may avoid converting NaNs in the eval_populations, and catch the
#   NaNs as late as possible (in the algorithm itself). We can give the
#   user the possibility of triggering a warning or error when a NaN is
#   encountered, if she's sure her primitives shouldn't EVER produce
#   them
def eval_populations(
    input: torch.Tensor,
    genomes: torch.Tensor,
    primitive_arities: torch.Tensor,
    n_inputs: int,
    n_outputs: int,
    n_hidden: int,
    primitive_functions: Iterable[Primitive],
    max_arity: Union[int, torch.Tensor, None] = None,
    nan_to_zero: bool = True,  # replace NaN with 0. in the output
):
    device = input.device
    if max_arity is None:
        max_arity = torch.max(primitive_arities)
    n_populations, n_individuals = genomes.size(0), genomes.size(1)
    output_start, output_end = n_inputs + n_hidden, n_inputs + n_hidden + n_outputs
    output_nodes = torch.arange(output_start, output_end, device=device)
    populations, individuals, nodes = torch.meshgrid(
        torch.arange(n_populations, device=device),
        torch.arange(n_individuals, device=device),
        output_nodes,
        indexing="ij",
    )
    populations = populations.flatten()
    individuals = individuals.flatten()
    nodes = nodes.flatten()
    output = eval_nodes(
        input=input,
        populations=populations,
        individuals=individuals,
        nodes=nodes,
        genomes=genomes,
        primitive_arities=primitive_arities,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_functions=primitive_functions,
        max_arity=max_arity,
    )
    single_input_size = input.size()[1:]
    if nan_to_zero:
        output[output.isnan()] = 0.0
    return output.reshape(n_populations, n_individuals, *single_input_size)


def eval_nodes(
    input: torch.Tensor,
    populations: torch.Tensor,
    individuals: torch.Tensor,
    nodes: torch.Tensor,
    genomes: torch.Tensor,
    primitive_arities: torch.Tensor,
    n_inputs: int,
    n_outputs: int,
    n_hidden: int,
    max_arity: Union[int, torch.Tensor],
    primitive_functions: Iterable[Primitive],
):
    device = input.device
    output_size = [nodes.size(0), *(input.size()[1:])]
    output = torch.zeros(output_size, dtype=input.dtype, device=device)
    is_input_node = nodes < n_inputs
    is_output_node = nodes >= n_inputs + n_hidden
    is_hidden_node = ~is_input_node & ~is_output_node
    if torch.any(is_input_node):
        output[is_input_node] = input[nodes[is_input_node]]
    if torch.any(is_output_node):
        conn_loci = 1 + (1 + max_arity) * nodes[is_output_node]
        conn_genes = genomes[populations, individuals, conn_loci]
        output[is_output_node] = eval_nodes(
            input=input,
            populations=populations,
            individuals=individuals,
            nodes=conn_genes,
            genomes=genomes,
            primitive_arities=primitive_arities,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            max_arity=max_arity,
            primitive_functions=primitive_functions,
        )
    if not torch.any(is_hidden_node):
        return output
    # We only need to compute hidden node outputs from here on.
    populations = populations[is_hidden_node]
    individuals = individuals[is_hidden_node]
    nodes = nodes[is_hidden_node]
    func_loci = (1 + max_arity) * nodes
    func_genes = genomes[populations, individuals, func_loci]
    arities = primitive_arities[func_genes]
    conn_loci = aranges(1 + func_loci, 1 + func_loci + arities)
    populations = torch.repeat_interleave(populations, dim=0, repeats=arities)
    individuals = torch.repeat_interleave(individuals, dim=0, repeats=arities)
    conn_genes = genomes[populations, individuals, conn_loci]
    hidden_input = eval_nodes(
        input=input,
        populations=populations,
        individuals=individuals,
        nodes=conn_genes,
        genomes=genomes,
        primitive_arities=primitive_arities,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        max_arity=max_arity,
        primitive_functions=primitive_functions,
    )
    output[is_hidden_node] = eval_primitives(
        inputs=hidden_input,
        primitives=func_genes,
        primitive_arities=primitive_arities,
        primitive_functions=primitive_functions,
    )
    return output


def eval_primitives(
    inputs: torch.Tensor,
    primitives: torch.Tensor,
    primitive_arities: torch.Tensor,
    primitive_functions: Iterable[Primitive],
):
    input_size = inputs.size()[1:]
    device = inputs.device
    output = torch.zeros([primitives.size(0), *input_size], device=device)
    next_first_inputs = primitive_arities[primitives].cumsum(dim=0)
    for i, f in enumerate(primitive_functions):
        mask = primitives == i
        n_calls = mask.count_nonzero()
        # TODO this causes dev-host sync(?)
        if n_calls == 0:
            continue
        arity = primitive_arities[i]
        ends = next_first_inputs[mask]
        starts = ends - arity
        prim_inputs = inputs[aranges(starts, ends)]
        # A simple reshape into (arity, n_calls) doesn't work since it would be
        # "transposed", so we reshape into (n_calls, arity) and then swap the
        # first 2 dimensions.
        # TODO this doesn't typecheck, but it works smh
        prim_inputs = prim_inputs.reshape(n_calls, arity, *input_size)
        # movedim(1, 0) also works
        prim_inputs = prim_inputs.transpose(1, 0)
        prim_outputs = f(prim_inputs)
        prim_outputs = prim_outputs.reshape(n_calls, *input_size)
        output[mask] = prim_outputs
    return output


def count_alleles(
    n_inputs: int,
    n_outputs: int,
    n_hidden: int,
    n_primitives: int,
    max_arity: int,
    dtype: Optional[torch.dtype] = None,  # long
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    dtype = dtype or torch.long
    genes_per_node = 1 + max_arity
    n_nodes = n_inputs + n_hidden + n_outputs
    # TODO use torch.repeat_interleave?
    n_alleles = torch.ones(genes_per_node, n_nodes, dtype=dtype, device=device)
    connection_alleles = torch.arange(n_nodes, dtype=dtype, device=device)
    n_alleles = n_alleles * connection_alleles
    n_alleles = n_alleles.T.reshape(genes_per_node * n_nodes)
    k = genes_per_node
    n_alleles[k * n_inputs : k * (n_inputs + n_hidden) : k] = n_primitives
    n_alleles[0 : k * n_inputs] = 1
    n_alleles[k * (n_inputs + n_hidden) : k * n_nodes] = 1
    n_alleles[1 + k * (n_inputs + n_hidden) : 1 + k * n_nodes : k] = n_inputs + n_hidden
    return n_alleles


# device will be n_alleles.device
def random_populations(
    n_populations: int,
    pop_size: int,
    n_alleles: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,  # randints default
) -> torch.Tensor:
    genome_size = n_alleles.size(0)
    # TODO avoid generating input region and non-coding output genes
    genomes = randints(
        highs=n_alleles,
        size=[n_populations, pop_size, genome_size],
        generator=generator,
        dtype=dtype,
    )
    return genomes


# extract random alternatives for each gene in loci (must be a bool mask for now)
def random_alternative_alleles(
    genomes: torch.Tensor,
    loci: torch.Tensor,  # has to be of dtype bool for now
    n_alleles: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,  # dnas.dtype
) -> torch.Tensor:
    dtype = dtype or genomes.dtype
    sups: torch.Tensor
    assert loci.dtype is torch.bool
    sups = torch.masked_select(n_alleles, loci) - 1
    alts = randints(highs=sups, generator=generator, dtype=dtype)
    alts[alts >= genomes[loci]] += 1
    return alts


def mutate(
    genomes: torch.Tensor,
    rate: float,
    n_alleles: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    in_place: bool = False,
) -> torch.Tensor:
    """TODO document the mutate function"""

    loci = random_mask_like(
        genomes, rate=rate, generator=generator, device=n_alleles.device
    )
    # Filter loci without alternatives.
    loci = loci & (n_alleles > 1)
    mutated_genes = random_alternative_alleles(
        genomes=genomes, loci=loci, n_alleles=n_alleles, generator=generator
    )
    if in_place:
        genomes[loci] = mutated_genes
        return genomes
    else:
        mutated_dnas = genomes.clone()
        mutated_dnas[loci] = mutated_genes
        return mutated_dnas
