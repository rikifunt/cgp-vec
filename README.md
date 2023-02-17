# `cgp-vec`: Vectorized Cartesian Genetic Programming

`cgp-vec` is a small library implementing Cartesian Genetic Programming
(CGP) parallelized over multiple independent populations, using PyTorch
as a backend for basic vectorized operations.

Since this library is still in a very early development stage, the API
may vary from commit to commit, together with the accuracy and
completeness of the documentation.

## Benchmarks

More detailed benchmarks will be available soon. Testing the examples on
a Google Colab GPU runtime led to execution times in the order of
minutes for 1000 runs of 500 evolution steps of CGP on Koza symbolic
regression problems, with configurations similar to those in [this GECCO
'07 paper](https://dl.acm.org/doi/10.1145/1276958.1277276), which uses
bigger populations than most CGP literature.

## Development status

The library is still in a very early development stage. The provided
functionality works as intended, and is sufficient to implement common
CGP experiments; however, for now it only supports CGP configurations
with 1-row genotypes and no "levels back" parameter. This is the most
common CGP configuration, since it can encode any (bounded) directed
acyclic graph. Some minor unit tests are also missing.

## Planned features

- support for all classical CGP configurations (multiple rows, levels
  back parameter);

- explicit generation of efficient (vectorized) phenotypes, in the form
  of either python callables or PyTorch modules;

- notable CGP variants from J. Miller's book "Cartesian Genetic
  Programming";

- convenience functions (vectorized) for common CGP problems: symbolic
  regression losses, etc.

## Installation

While this repository contains a proper Python package, it is not yet
registered on PyPy, so it must be installed via cloning and `pip install
.` for now; the only external dependency is the PyTorch library. This
library has only been tested on the major Python versions 3.7 and 3.10.

Unit tests are in the `tests` directory; to run all of them, one can run
`python -m unittest -v "tests.test_cgpv"` at the top directory. 

## Usage

An overview of the API is given below (also see
[Documentation](#documentation)). Full symbolic regression examples are
also available in the `examples` directory, both as python scripts and
as notebooks.

TODO introduce the new API (with an example?)

## Documentation

Documentation can be generated in various format with
[Sphinx](https://www.sphinx-doc.org/en/master/#), using `docs/Makefile`
or `docs/make.bat`. For example, to generate the documentation in HTML
format, one can run:

```
cd docs
make html
```

`docs/requirements.txt` contains the packages needed to build the
documentation.

## Contributing

`CONTRIBUTING.md` is the starting point for all information concerning
the development of this library, both for code and documentation.

## BibTeX Citation

```
@misc{cgpvec-2022-git,
    author = {Fanti, Andrea and Gallotta, Roberto},
    title = {{cgp-vec}: Vectorized Cartesian Genetic Programming},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository}}
}
```
