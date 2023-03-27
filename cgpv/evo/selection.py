from typing import Optional, Tuple

import torch


def combinations(
    items: torch.Tensor,  # (N, M, P)
    n_combinations: int,  # T
    k: int,  # S
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (N, T, S, P) and (N, T, S)
    """Extract multiple random combinations from a tensor of items.

    Given a tensor of items of shape (N, M, P), with each item being a
    vector of size P, extract n_combinations combinations, each of size
    k, for each row of items.

    A tuple of 2 tensors is returned:
    - the extracted combinations, of shape (N, n_combinations, k, P);
    - the indices of the items extracted, of shape
      (N, n_combinations, k).

    In other words, the first tensor returned by this operation is a
    tensor y such that y[i][j] is a sequence of k items from items[i],
    extracted uniformly *without* replacement, for all j in
    {0, ..., n_combinations}. The second tensor returned is a tensor z
    such that z[i][j] contains the indices in {0, M} of the items
    contained in y[i][j].
    """
    # "vectorized randperm", inspired by https://discuss.pytorch.org/t/what-is-the-most-efficient-way-to-shuffle-each-row-of-a-tensor-with-different-shuffling-order-for-each-of-the-row/109772
    shuffled_indices = torch.argsort(
        torch.rand(
            [items.size(0), n_combinations, items.size(1)],
            generator=generator,
        ),
        dim=2,
    )  # (N, T, M)
    extracted_indices = shuffled_indices[..., :k]  # (N, T, S)
    extracted_items = torch.gather(
        items.unsqueeze(1).expand(
            -1, extracted_indices.size(1), -1, -1
        ),  # (N, T, M, P)
        dim=2,
        index=extracted_indices.unsqueeze(3).expand(
            -1, -1, -1, items.size(2)
        ),  # (N, T, S, P)
    )  # (N, T, S, P)
    return extracted_items, extracted_indices


def tournaments(
    items: torch.Tensor,  # (N, M, P)
    scores: torch.Tensor,  # (N, M)
    n_tournaments: int,  # T
    tournament_size: int,  # S
    minimize: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (N, T, P) and (N, T)
    """Perform multiple random tournaments among the given items.

    Given a tensor of items of shape (N, M, P), perform n_tournaments of
    size tournament_size using the given tensor of scores of shape
    (N, M). A tuple of 2 tensor is returned:
    - the winners, of shape (N, n_tournaments, P);
    - the scores of the winners, of shape (N, n_tournaments).

    Each tournament is performed by extracting tournament_size distinct
    participants at random, and then comparing their scores to find the
    winner. If minimize is False (default), the winner is the item with
    the maximum score; otherwise, the winner is the item with the
    minimum score.

    In other words, the first tensor returned is a tensor y such that
    y[i][j] is the winner of the (j+1)-th tournament performed on items
    in items[i]. The second tensor
    returned is a tensor z such that z[i][j] is the score of the winner
    of the (j+1)-th tournament performed on items in items[i].
    """
    participants, participant_indices = combinations(
        items=items,
        n_combinations=n_tournaments,
        k=tournament_size,
        generator=generator,
    )  # (N, T, S, P) and (N, T, S)
    participant_scores = torch.gather(
        scores.unsqueeze(1).expand(-1, n_tournaments, -1),
        dim=2,
        index=participant_indices,
    )  # (N, T, S)
    reduce = torch.max if not minimize else torch.min
    winner_scores, winner_indices = reduce(
        participant_scores, dim=2
    )  # (N, T) and (N, T)
    winners = torch.gather(
        participants,
        dim=2,
        index=winner_indices.unsqueeze(2)
        .unsqueeze(2)
        .expand(-1, -1, -1, participants.size(3)),  # (N, T, 1, P)
    ).squeeze(
        2
    )  # (N, T, P)
    return winners, winner_scores


# assumes weights are positive and sum to 1 along rows
# optimized for when there is only 1 item in each row: simply returns it
# n_rounds times
def roulette_wheel(
    n_rounds,
    items: torch.Tensor,
    weights: torch.Tensor,
    normalize_weights: bool = True,
    generator: Optional[torch.Generator] = None,
):
    device = items.device
    rows = torch.arange(weights.size(0), device=device)
    rows = rows.reshape(-1, 1).tile(1, n_rounds)
    if items.size(1) == 1:
        columns = torch.zeros_like(rows)
        return items[rows, columns]
    if normalize_weights:
        sums = weights.sum(dim=1, keepdim=True)
        if torch.any(sums == 0.0):
            raise ValueError("roulette_wheel: 0-sum weights for some populations(s)")
        weights = weights / sums
    # shape: (n_rows, n_cols)
    cdfs = weights.cumsum(dim=1)
    thresholds = torch.rand(
        *[n_rounds, weights.size(0), 1], generator=generator, device=device
    )
    # (cdfs < thresholds) has shape (n_rounds, n_pops, n_inds)
    columns = (cdfs < thresholds).sum(dim=2).T
    # columns has shape (n_rounds, n_pops)
    # TODO check if this can be simplified (see columns version)
    selected = items[rows, columns]
    return selected


# assumes weights are positive and sum to 1 over columns
def roulette_wheel_columns(
    n_rounds,
    items: torch.Tensor,
    weights: torch.Tensor,
    generator: Optional[torch.Generator] = None,
):
    device = items.device
    s = weights.cumsum(dim=0)
    r = torch.rand(n_rounds, 1, weights.size(1), generator=generator, device=device)
    # s has shape (n_inds, n_pops)
    # (s < r) has shape ( n_rounds, n_inds, n_pops )
    k = (s < r).sum(dim=1)
    # k has shape ( n_rounds, n_pops )
    return items[k, torch.arange(weights.size(1), device=device)]


def plus_selection(
    genomes: torch.Tensor,
    fitnesses: torch.Tensor,
    offspring: torch.Tensor,
    offspring_fitnesses: torch.Tensor,
    # TODO also support offspring_first: bool = True,
    descending: bool = True,
):
    device = genomes.device
    combined = torch.hstack([offspring, genomes])
    combined_fitnesses = torch.hstack([offspring_fitnesses, fitnesses])
    sorted_fitnesses, sorted_indices = torch.sort(
        combined_fitnesses, dim=1, descending=descending, stable=True
    )
    n = genomes.size(1)
    topk_fitnesses = sorted_fitnesses[:, 0:n]
    topk_indices = sorted_indices[:, 0:n]
    populations = torch.arange(genomes.size(0), device=device)
    populations = populations.reshape(-1, 1).tile(1, n)
    topk = combined[populations, topk_indices]
    return topk, topk_fitnesses
