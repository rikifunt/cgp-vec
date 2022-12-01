from typing import Any, Union, Tuple, List, Dict, Optional, Iterable, Callable

import torch

# TODO generalize to any size of starts/stops, now works only with 1D tensors
def aranges(
    starts: torch.Tensor,
    stops: torch.Tensor,
    dtype: Optional[torch.dtype] = None,  # arange default (long)
) -> torch.Tensor:
    device = starts.device
    lengths = stops - starts
    steps = torch.repeat_interleave(stops - lengths.cumsum(dim=0), lengths)
    total_length = lengths.sum()
    # TODO this doesn't typecheck, but it works smh
    return steps + torch.arange(total_length, dtype=dtype, device=device)


def randints(
    highs: torch.Tensor,
    size: Optional[Any] = None,  # anything that can be used as torch size
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,  # long
) -> torch.Tensor:
    size = size if size is not None else highs.size()
    dtype = dtype or torch.long
    normalized = torch.rand(*size, generator=generator, device=highs.device)
    inflated = normalized * highs
    return inflated.to(dtype)


def randints_like(
    other: torch.Tensor,
    highs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return randints(highs=highs, size=other.size(), generator=generator, dtype=dtype)


def random_mask(
    size: Any,  # anything that can be used as torch size
    rate: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    thresholds = torch.rand(size, generator=generator, device=device)
    return thresholds < rate


def random_mask_like(
    other: torch.Tensor,
    rate: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,  # device of other
) -> torch.Tensor:
    return random_mask(
        other.size(), rate=rate, generator=generator, device=device or other.device
    )


def seeded_generator(seed: int, *init_args, **init_kwargs) -> torch.Generator:
    rng = torch.Generator(*init_args, **init_kwargs)
    rng.manual_seed(seed)
    return rng
