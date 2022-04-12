
from typing import Any, Union, Tuple, List, Dict, Optional, Iterable, Callable

import torch

from .utils import randints, random_mask_like

# Build a 1D tensor that at each locus tells how many alleles that locus admits.
def count_alleles(
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        n_primitives: int,
        max_arity: int,
        dtype: Optional[torch.dtype] = None, # long
        device: Optional[torch.device] = None
      ) -> torch.Tensor:
    dtype = dtype or torch.long
    genes_per_node = 1 + max_arity
    n_nodes = n_inputs + n_hidden + n_outputs
    #TODO use torch.repeat_interleave?
    n_alleles = torch.ones(genes_per_node, n_nodes, dtype=dtype, device=device)
    connection_alleles = torch.arange(n_nodes, dtype=dtype, device=device)
    n_alleles = n_alleles * connection_alleles
    n_alleles = n_alleles.T.reshape(genes_per_node*n_nodes)
    k = genes_per_node
    n_alleles[k*n_inputs : k*(n_inputs+n_hidden) : k] = n_primitives
    n_alleles[0 : k*n_inputs] = 1
    n_alleles[k*(n_inputs+n_hidden) : k*n_nodes] = 1
    n_alleles[1 + k*(n_inputs+n_hidden) : 1 + k*n_nodes : k ] \
        = n_inputs + n_hidden
    return n_alleles

# device will be n_alleles.device
def random_populations(
        n_populations: int,
        pop_size: int,
        n_alleles: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None, # randints default
      ) -> torch.Tensor:
    dna_size = n_alleles.size(0)
    #TODO is there a way to avoid generating input region and non-coding output
    #genes?
    dnas = randints(
        highs=n_alleles, size=[n_populations, pop_size, dna_size],
        generator=generator, dtype=dtype
      )
    return dnas

# extract random alternatives for each gene in loci (must be a bool mask for now)
# dnas, loci and n_alleles must be on the same device
def random_alternative_alleles(
        dnas: torch.Tensor,
        loci: torch.Tensor, # has to be of dtype bool for now
        n_alleles: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None # dnas.dtype
      ) -> torch.Tensor:
    dtype = dtype or dnas.dtype
    sups: torch.Tensor
    assert(loci.dtype is torch.bool)
    sups = torch.masked_select(n_alleles, loci) - 1
    alts = randints(highs=sups, generator=generator, dtype=dtype)
    alts[alts >= dnas[loci]] += 1
    return alts

def mutate(
        dnas: torch.Tensor,
        rate: float,
        n_alleles: torch.Tensor,
        generator: torch.Generator = None,
        in_place: bool = False, # if true, mutates dnas directly and returns it
      ) -> torch.Tensor:
    loci = random_mask_like(dnas, rate=rate, generator=generator,
                            device=n_alleles.device)
    # Filter loci without alternatives.
    loci = loci & (n_alleles > 1)
    mutated_genes = random_alternative_alleles(
        dnas=dnas, loci=loci, n_alleles=n_alleles, generator=generator
      )
    if in_place:
        dnas[loci] = mutated_genes
        return dnas
    else:
        mutated_dnas = dnas.clone()
        mutated_dnas[loci] = mutated_genes
        return mutated_dnas