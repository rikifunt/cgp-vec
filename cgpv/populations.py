from typing import (
    Any,
    Union,
    Tuple,
    List,
    Dict,
    Optional,
    Iterable,
    Callable,
    NamedTuple,
)
from types import ModuleType

import torch

import cgpv


# Methods that return a new Populations object usually internally share the
# configuration or the dnas with the original object, i.e. the two object are
# "coupled".
# Methods that return or assign attributes of a Populations object make no
# effort to decouple it from the object, so the user must take care not to
# accidentally invalidate the state by modifying the attribute outside of the
# object.
class Populations:

    # TODO the dtype for n_alleles is not constrained to be long int (it is not
    # used for indexing); cleanly allow the user to specify it

    @staticmethod
    def random(
        n_populations: int,
        pop_size: int,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        primitive_functions: List[cgpv.PrimitiveType],
        primitive_arities: Union[List[int], torch.Tensor],
        descending_fitness: bool = True,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> "Populations":
        n_primitives = len(primitive_functions)
        max_arity = max(primitive_arities)
        n_alleles = cgpv.count_alleles(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            n_primitives=n_primitives,
            max_arity=max_arity,
            dtype=torch.long,
            device=device,
        )
        dnas = cgpv.random_populations(
            n_populations=n_populations,
            pop_size=pop_size,
            n_alleles=n_alleles,
            generator=generator,
            dtype=torch.long,
        )
        return Populations(
            dnas=dnas,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden,
            primitive_functions=primitive_functions,
            primitive_arities=primitive_arities,
            descending_fitness=descending_fitness,
            n_alleles=n_alleles,
            n_primitives=n_primitives,
            max_arity=max_arity,
            device=device,
        )

    # This constructor is tailored to allow writing efficient operators on
    # Populations objects: it has some redundant parameters that, when
    # passed, avoid recomputing possibly known information (e.g. the genome
    # configuration); moreover, no effort is made to decouple the object from
    # the given parameters, so that the constructed object is always considered
    # coupled.
    def __init__(
        self,
        dnas: torch.Tensor,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        primitive_functions: List[cgpv.PrimitiveType],
        primitive_arities: Union[List[int], torch.Tensor],
        descending_fitness: bool = True,
        # From here ---
        n_alleles: Optional[Union[List[int], torch.Tensor]] = None,
        n_primitives: Optional[int] = None,
        max_arity: Optional[int] = None,
        # --- to here, parameters are redundant.
        device: Optional[torch.device] = None,
    ) -> None:
        self._dnas = dnas
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_hidden = n_hidden
        self._primitive_functions = primitive_functions
        self._n_primitives = (
            len(primitive_functions) if n_primitives is None else n_primitives
        )
        self._max_arity = max(primitive_arities) if max_arity is None else max_arity
        if isinstance(primitive_arities, list):
            self._primitive_arities = torch.tensor(primitive_arities, device=device)
        else:
            self._primitive_arities = primitive_arities
        self._descending_fitness = descending_fitness
        if n_alleles is None:
            self._n_alleles = cgpv.count_alleles(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_hidden=n_hidden,
                n_primitives=self._n_primitives,
                max_arity=self._max_arity,
                dtype=torch.long,
                device=device,
            )
        elif isinstance(n_alleles, list):
            self._n_alleles = torch.tensor(n_alleles, device=device)
        else:
            self._n_alleles = n_alleles
        self._device = device
        self._fitnesses: Optional[torch.Tensor] = None

    @property
    def dnas(self) -> torch.Tensor:
        return self._dnas

    @dnas.setter
    def dnas(self, dnas: torch.Tensor) -> None:
        self._dnas = dnas

    # convenience class for copying, contains everything but the dnas
    class Configuration(NamedTuple):
        n_inputs: int
        n_outputs: int
        n_hidden: int
        primitive_functions: List[cgpv.PrimitiveType]
        primitive_arities: Union[List[int], torch.Tensor]
        descending_fitness: bool
        n_alleles: Optional[Union[List[int], torch.Tensor]]
        n_primitives: Optional[int]
        max_arity: Optional[int]
        device: Optional[torch.device]

    def configuration(self) -> "Populations.Configuration":
        return Populations.Configuration(
            n_inputs=self._n_inputs,
            n_outputs=self._n_outputs,
            n_hidden=self._n_hidden,
            primitive_functions=self._primitive_functions,
            primitive_arities=self._primitive_arities,
            descending_fitness=self._descending_fitness,
            n_alleles=self._n_alleles,
            n_primitives=self._n_primitives,
            max_arity=self._max_arity,
            device=self._device,
        )

    def _validate_number(self, x, expected, name: str, raise_: bool):
        if x == expected:
            return True
        if raise_:
            raise ValueError(f"Wrong {name} ({x}), expected {expected}")

    # TODO allow to either print tensors, or write them to file
    def _validate_tensor(self, x, expected, name: str, raise_: bool):
        if torch.equal(x, expected):
            return True
        if raise_:
            raise ValueError(f"Wrong {name}")

    # TODO also expose this in the functional interface
    def validate(self, raise_: bool = True) -> bool:

        if not self._validate_number(
            len(self._primitive_functions),
            expected=self._n_primitives,
            name="len(primitive_functions)",
            raise_=raise_,
        ):
            return False
        if not self._validate_number(
            self._primitive_arities.size(0),
            expected=self._n_primitives,
            name="primitive_arities.size(0)",
            raise_=raise_,
        ):
            return False

        if not self._validate_number(
            self._max_arity,
            max(self._primitive_arities),
            name="max_arity",
            raise_=raise_,
        ):
            return False

        dna_size = self._dnas.size(2)
        n_nodes = self._n_inputs + self._n_outputs + self._n_hidden

        if not self._validate_number(
            dna_size,
            expected=n_nodes * (self._max_arity + 1),
            name="DNA size",
            raise_=raise_,
        ):
            return False

        if not self._validate_number(
            self._n_alleles.size(0),
            expected=dna_size,
            name="n_alleles.size(0)",
            raise_=raise_,
        ):
            return False
        expected_n_alleles = cgpv.count_alleles(
            n_inputs=self._n_inputs,
            n_outputs=self._n_outputs,
            n_hidden=self._n_hidden,
            n_primitives=self._n_primitives,
            max_arity=self._max_arity,
            dtype=self._n_alleles.dtype,
            device=self._n_alleles.device,
        )
        if not self._validate_tensor(
            self._n_alleles,
            expected=expected_n_alleles,
            name="n_alleles",
            raise_=raise_,
        ):
            return False

        if not torch.all(self._dnas >= 0) and torch.all(self._dnas < self._n_alleles):
            if raise_:
                raise ValueError(f"DNAs contain invalid alleles")
            return False

        # TODO check device is correct
        # TODO check fitnesses are sane and of correct shape
        return True

    def mutate(
        self, rate: float, generator: Optional[torch.Generator], in_place: bool = False
    ) -> "Populations":
        mutated_dnas = cgpv.mutate(
            dnas=self._dnas,
            rate=rate,
            n_alleles=self._n_alleles,
            generator=generator,
            in_place=in_place,
        )
        if in_place:
            return self
        conf = self.configuration()
        return Populations(dnas=mutated_dnas, **conf._asdict())

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return cgpv.eval_populations(
            input=input,
            dnas=self._dnas,
            n_inputs=self._n_inputs,
            n_outputs=self._n_outputs,
            n_hidden=self._n_hidden,
            primitive_arities=self._primitive_arities,
            max_arity=self._max_arity,
            primitive_functions=self._primitive_functions,
        )

    @property
    def fitnesses(self):
        if self._fitnesses is None:
            raise ValueError("Fitnesses not yet assigned.")
        return self._fitnesses

    # for now, fitnesses should always be non-negative
    @fitnesses.setter
    def fitnesses(self, fitnesses: torch.Tensor):
        self._fitnesses = fitnesses

    # invalid if fitnesses have not been assigned
    def roulette_wheel(
        self, n_rounds: int, generator: Optional[torch.Generator] = None
    ) -> "Populations":
        weights = self.fitnesses
        if not self._descending_fitness:
            weights = 1.0 / weights  # TODO is adding a small eps here a good idea?
        extracted_dnas = cgpv.roulette_wheel(
            n_rounds,
            items=self._dnas,
            weights=weights,
            normalize_weights=True,
            generator=generator,
        )
        conf = self.configuration()
        return Populations(dnas=extracted_dnas, **conf._asdict())

    # invalid if fitnesses have not been assigned
    def tournament(self, n_winners: int) -> "Populations":
        if self._fitnesses is None:
            raise ValueError("Fitnesses have not been computed yet")
        winner_dnas, winner_fitnesses = cgpv.tournament(
            n_winners=n_winners,
            items=self._dnas,
            scores=self._fitnesses,
            descending=self._descending_fitness,
            return_scores=True,
        )
        conf = self.configuration()
        winners = Populations(dnas=winner_dnas, **conf._asdict())
        winners.fitnesses = winner_fitnesses
        return winners

    # always prioritizes the object it is called on, assumes both populations
    # have fitnesses; the configuration of the other object is ignored
    def plus_selection(self, other: "Populations") -> "Populations":
        selected_dnas, selected_fitnesses = cgpv.plus_selection(
            parents=other.dnas,
            parent_fitnesses=other.fitnesses,
            offspring=self._dnas,
            offspring_fitnesses=self.fitnesses,
            descending=self._descending_fitness,
        )
        conf = self.configuration()
        selected = Populations(dnas=selected_dnas, **conf._asdict())
        selected.fitnesses = selected_fitnesses
        return selected

    def comma_selection(self):
        raise NotImplementedError("Comma selection not implemented yet :(")
