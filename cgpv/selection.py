
from typing import Any, Union, Tuple, List, Dict, Optional, Iterable, Callable

import torch


#TODO for all selection functions also support returning indices and/or weights/scores/etc.

# assumes weights are positive and sum to 1 along rows
# optimized for when there is only 1 item in each row: simply returns it
# n_rounds times
def roulette_wheel(
        n_rounds,
        items: torch.Tensor,
        weights: torch.Tensor,
        normalize_weights: bool = True,
        generator: Optional[torch.Generator] = None
      ):
    device = items.device
    rows = torch.arange(weights.size(0), device=device)
    rows = rows.reshape(-1, 1).tile(1, n_rounds)
    if items.size(1) == 1:
        columns = torch.zeros_like(rows)
        return items[rows, columns]
    if normalize_weights:
        sums = weights.sum(dim=1, keepdim=True)
        if torch.any(sums == 0.):
            raise ValueError('roulette_wheel: 0-sum weights for some '
                             'populations(s)')
        weights = weights / sums
    # shape: (n_rows, n_cols)
    cdfs = weights.cumsum(dim=1)
    thresholds = torch.rand(*[n_rounds, weights.size(0), 1],
                            generator=generator, device=device)
    # (cdfs < thresholds) has shape (n_rounds, n_pops, n_inds)
    columns = (cdfs < thresholds).sum(dim=2).T
    # columns has shape (n_rounds, n_pops)
    #TODO check if this mess can be simplified (see columns version)
    selected = items[rows, columns]
    return selected

# assumes weights are positive and sum to 1 over columns
def roulette_wheel_columns(
        n_rounds,
        items: torch.Tensor,
        weights: torch.Tensor,
        normalize_weights: bool = True,
        generator: Optional[torch.Generator] = None
      ):
    device = items.device
    s = weights.cumsum(dim=0)
    r = torch.rand(n_rounds, 1, weights.size(1),
                   generator=generator, device=device)
    # s has shape (n_inds, n_pops)
    # (s < r) has shape ( n_rounds, n_inds, n_pops )
    k = (s < r).sum(dim=1)
    # k has shape ( n_rounds, n_pops )
    # 
    return items[k, torch.arange(weights.size(1), device=device)]

# this has no stability guarantee as it calls torch.topk
def tournament(
        n_winners: int,
        items: torch.Tensor,
        scores: torch.Tensor,
        descending: bool = True,
        return_scores: bool = False
      ):
    device = items.device
    best_scores, best_indices = torch.topk(scores, k=n_winners, dim=1,
                                           largest=descending)
    rows = torch.arange(items.size(0), device=device)
    rows = rows.reshape(-1, 1).tile(1, n_winners)
    winners = items[rows, best_indices]
    if return_scores:
        return winners, best_scores
    else:
        return winners

def plus_selection(
        parent_fitnesses: torch.Tensor,
        offspring_fitnesses: torch.Tensor,
        parents: torch.Tensor,
        offspring: torch.Tensor,
        #TODO also support offspring_first: bool = True,
        descending: bool = True,
      ):
    device = parents.device
    combined = torch.hstack([offspring, parents])
    combined_fitnesses = torch.hstack([offspring_fitnesses, parent_fitnesses])
    sorted_fitnesses, sorted_indices = torch.sort(
        combined_fitnesses, dim=1, descending=descending, stable=True
      )
    k = parents.size(1)
    topk_fitnesses = sorted_fitnesses[:, 0:k]
    topk_indices = sorted_indices[:, 0:k]
    populations = torch.arange(parents.size(0), device=device)
    populations = populations.reshape(-1, 1).tile(1, k)
    topk = combined[populations, topk_indices]
    return topk, topk_fitnesses
