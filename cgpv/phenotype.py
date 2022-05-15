from typing import Any, Union, Tuple, List, Dict, Optional, Iterable, Callable

import torch

from cgpv.utils import aranges


PrimitiveType = Callable[[torch.Tensor], torch.Tensor]


def eval_populations(
        input: torch.Tensor,
        dnas: torch.Tensor,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        primitive_arities: torch.Tensor,
        max_arity: int,
        primitive_functions: List[Callable[[torch.Tensor], torch.Tensor]],
        nan_to_zero: bool = True # replace NaN with 0. in the output
      ):
    device = input.device
    n_populations, n_individuals = dnas.size(0), dnas.size(1)
    output_start, output_end = n_inputs+n_hidden, n_inputs+n_hidden+n_outputs
    output_nodes = torch.arange(output_start, output_end, device=device)
    populations, individuals, nodes \
        = torch.meshgrid(
            torch.arange(n_populations, device=device),
            torch.arange(n_individuals, device=device),
            output_nodes,
            indexing='ij'
          )
    populations = populations.flatten()
    individuals = individuals.flatten()
    nodes = nodes.flatten()
    output = eval_nodes(
        input=input,
        populations=populations,
        individuals=individuals,
        nodes=nodes,
        dnas=dnas,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_arities=primitive_arities,
        primitive_functions=primitive_functions,
        max_arity=max_arity 
      )
    single_input_size = input.size()[1:]
    if nan_to_zero:
        output[output.isnan()] = 0.
    return output.reshape(n_populations, n_individuals, *single_input_size)


def eval_nodes(
        input: torch.Tensor,
        populations: torch.Tensor,
        individuals: torch.Tensor,
        nodes: torch.Tensor,
        dnas: torch.Tensor,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        primitive_arities: torch.Tensor,
        max_arity: int,
        primitive_functions: List[Callable[[torch.Tensor], torch.Tensor]]
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
        conn_loci = 1 + (1+max_arity)*nodes[is_output_node]
        conn_genes = dnas[populations, individuals, conn_loci]
        output[is_output_node] = eval_nodes(
            input=input,
            populations=populations,
            individuals=individuals,
            nodes=conn_genes,
            dnas=dnas,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            primitive_arities=primitive_arities,
            max_arity=max_arity,
            primitive_functions=primitive_functions
          )
    if not torch.any(is_hidden_node):
        return output
    # We only need to compute hidden node outputs from here on.
    populations = populations[is_hidden_node]
    individuals = individuals[is_hidden_node]
    nodes = nodes[is_hidden_node]
    func_loci = (1+max_arity)*nodes
    func_genes = dnas[populations, individuals, func_loci]
    arities = primitive_arities[func_genes]
    conn_loci = aranges(1 + func_loci, 1 + func_loci + arities)
    populations = torch.repeat_interleave(populations, dim=0, repeats=arities)
    individuals = torch.repeat_interleave(individuals, dim=0, repeats=arities)
    conn_genes = dnas[populations, individuals, conn_loci]
    hidden_input = eval_nodes(
        input=input,
        populations=populations,
        individuals=individuals,
        nodes=conn_genes,
        dnas=dnas,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        primitive_arities=primitive_arities,
        max_arity=max_arity,
        primitive_functions=primitive_functions
      )
    output[is_hidden_node] = eval_primitives(
        inputs=hidden_input,
        primitives=func_genes,
        primitive_functions=primitive_functions,
        primitive_arities=primitive_arities
      )
    return output

def eval_primitives(
        inputs: torch.Tensor,
        primitives: torch.Tensor,
        primitive_functions,
        primitive_arities: torch.Tensor,
      ):
    input_size = inputs.size()[1:]
    device = inputs.device
    output = torch.zeros([primitives.size(0), *input_size], device=device)
    next_first_inputs = primitive_arities[primitives].cumsum(dim=0)
    for i, f in enumerate(primitive_functions):
        mask = primitives == i
        n_calls = mask.count_nonzero()
        #TODO this causes dev-host sync(?); use the shape instead?
        if n_calls == 0:
            continue
        arity = primitive_arities[i]
        ends = next_first_inputs[mask]
        starts = ends - arity
        ranges = aranges(starts, ends)
        prim_inputs = inputs[aranges(starts, ends)]
        # A simple reshape into (arity, n_calls) doesn't work since it would be
        # "transposed", so we reshape into (n_calls, arity) and then swap the
        # first 2 dimensions.
        #TODO this doesn't typecheck, but it works smh
        prim_inputs = prim_inputs.reshape(n_calls, arity, *input_size)
        # movedim(1, 0) also works
        prim_inputs = prim_inputs.transpose(1, 0)
        prim_outputs = f(prim_inputs)
        prim_outputs = prim_outputs.reshape(n_calls, *input_size)
        output[mask] = prim_outputs
    return output

#TODO write a custom kernel that evaluates multiple primitives in parallel?