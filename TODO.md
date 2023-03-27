This file contains the detailed development roadmap. Each header
contains the roadmap for a single task, i.e. a (possibly nested) list of
subtasks. Non-periodic subtasks have a checkbox specifying whether the
subtasks has been carried out or not. Tasks and subtasks can be numbered
to specify the order in which they should be carried out or their
relative priority. Important periodic tasks are usually placed at the
beginning, before any non-periodic task. When the library version is
updated, task and subtasks listed here that were completed during the
update are removed from here and their contents are added to the
changelog.


# Periodic maintainance

- (last update 30 nov 2022) update the library when colab switches to
  new versions of Python and/or PyTorch

	- update code that is affected by Python changes
	
	- update code that is affected by PyTorch changes
	
	- update the versions of the dependencies:
		
		- in `setup.cfg`
		
		- in `requirements.txt`


# 1: Finish setting up the repository infrastructure

1. [x] polish `requirements.txt` with only a minimal list of packages
   needed for development

2. [x] use `setup.cfg` instead of `setup.py`

3. [x] adopt the black code layout

	- [x] run black on the whole codebase

	- [x] add black to `requirements.txt`

3. [x] adopt sphinx for docs

	- [x] add sphinx to the `requirements.txt`

	- [x] write and test stubs for the user manual and API documentation

4. [x] document development process:

	3. [x] update workflow & code structure in `CONTRIBUTING.md` with:
		
		- [x] unit test workflow
		
		- [x] code linter invocation
		
		- [x] how to add documentation
	
	4. [x] add an introductory section to `CONTRIBUTING.md` which tells
	the reader where to look for dev stuff

	5. [x] update `README.md` mentioning `CONTRIBUTING.md`

5. [ ] write the basic API documentation

5. [ ] find a way to run python 3.11 on colab

	- remember to drop dependency on typing extensions when a Python
	version >= 3.8 is adopted (since we only use the `Protocol` feature)


# 2: API rehaul

0. [x] migrate to pytest for unit testing, since it supports markers and
   excluding tests easily

	- [x] migrate unittest code to pytest

	- [x] add test markers for slow tests and tests that need CUDA, and
	use them in existing tests

	- [x] test unit testing with pytest :)

	- [x] update relevant sections of `CONTRIBUTING.md`

1. [x] refactor package structure

	- [x] remove the `Populations` class

	- [x] move pytorch utilities to `cgpv.torch`

    - [x] move selection operators to `cgpv.evo.selection`

    - [x] move the classic cgp operators to `cgpv.cgp.classic`

2. [ ] add an extensible and out-of-the-box implementation of classic
   single-row CGP, such that the user can override the various steps
   (reproduction, selection, etc.) to produce other CGP variants

   - [x] add a CGP class that also stores intermediate tensors
     (offspring, offspring fitnesses, etc.)

   - [ ] split the run method of the CGP class into several methods
     that can be overridden in child classes to modify the corresponding
	 operators (mutation, selection, etc.)

   - [ ] add tests that check for correctness by matching known
     classic CGP performances (from another library, or from literature)

3. [ ] support 0-arity primitives

3. [ ] add the store args decorator and use it in existing code

4. [ ] add convenience functions that vectorize normal functions (or 
   functions vectorized over single populations) over multiple
   populations


# 3: Finish implementing basic CGP variants

1. [ ] implement (single-row) vectorized real crossover from gecco 07
   and a corresponding crossover-based reproduction operator for use in
   the EA/ES interface

2. [ ] implement other relevant crossover operators (vectorized,
   single-row) from CGP literature

2. [ ] implement multiple rows and levels back parameter


# 4: Major version release

1. [ ] implement the evotorch interface

1. [ ] add sensible utilities to the `cgpv.common` module:

	- TODO

1. [ ] identify, minimize and document dev-host synchronization points,
   e.g.:

	- passing tensors to Number parameters (e.g. for arange's end)

	- counting nonzero elements in tensors

1. [ ] support any python callable for which all arguments are tensors
   as a primitive (as of now, only callable with no keyword arguments
   are supported)

1. [ ] add minimal code examples, tutorials and user manual

1. [ ] switch to hatchling as build system?

1. [ ] consider putting `cgpv` in a `src` directory (see
   https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure%3E)

1. [ ] benchmark the library on Colab (and other platforms?), both on
   GPU and CPU

2. [ ] polish text files:

	- [ ] `CONTRIBUTING.md`
	
	- [ ] `README.md`

	- [ ] add an `env.yaml` for a conda environment for development
	
	- [ ] TODO others

3. [ ] do the actual release:

	- [ ] tag the commit etc.

	- [ ] register package on PyPi


# 5: Implement advanced phenotype generation features

- [ ] support the generation of torch module or python callable
  vectorized phenotypes

- [ ] support the generation of "layered" phenotypes by using residual
  connections


# Secondary tasks

- [ ] extend the interface of the GenomeTensor ABC implementing
  functions that are vectorized over any population shape, by reshaping
  from and to 2-dimensional populations

- [ ] add Genome and GenomeMatrix ABCs that provide genetic operators
  non-vectorized and vectorized over a single population, respectively

- [ ] implement notable CGP variants (see the CGP bible and more recent
  stuff)

- [ ] use a stable top-k algorithm for plus selection instead of stable
  sorting (to improve complexity)
