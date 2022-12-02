This file documents the workflow for releasing new versions of the
library and for committing code changes. As of now, it is still a
work-in-progress and many details are missing or yet to be determined.

For now, the library is developed on top of Python 3.7 to make sure it
is available on colab. As soon as a way to run Python 3.11 on colab is
found, the library will adopt Python 3.11.

# Contents

- [Development workflow](#development-workflow)

- [Coding guidelines](#coding-guidelines)

- [Unit testing](#unit-testing)

- [Writing documentation](#writing-documentation)

- [Codebase structure](#codebase-structure)


# Development workflow

`requirements.txt` contains a minimal set of Python packages needed for
the code development workflow.

## Working on new library versions

The workflow when developing a new version for the library is:

1. update the library version to be worked on in:

	- `setup.cfg`

2. work on all the commits (see below)

3. release: TODO

In the future, this workflow will probably be updated with git tagging
and branching.

## Individual commits

The workflow of a typical commit is:

0. choose what to work on (e.g. from `TODO.md`)

1. write code (see the [Coding guidelines](#coding-guidelines) below)

	- annotate types of everything, unless Python typing issues would
	make the code less readable
	
	- when modifying or extending the API:
	
		- add or modify unit tests (see [Adding or modifying unit
		tests](#adding-or-modifying-unit-tests))
	
		- document API modifications in (see [Writing
		documentation](#writing-documentation)):

			- the API documentation

			- the user manual (if it exists)
	
	- document the modification of internal functions:
		
		- add or modify docstrings of internal functions where needed
		(their name or behaviour is not self-explanatory)
		
		- add or modify the sections of `CONTRIBUTING.md` that talk
		about the modified internals

2. (optional) run the type checker (possibly interactively in your IDE)

	- suggested type checkers: 
	
		- [`mypy`](http://mypy-lang.org/)
		
		- Pyright (the type checker of the Pylance extension for VS
		code)

3. run the [`black`](https://github.com/psf/black) code linter, simply
   as `black <modified files>` or `black .` to run it on the whole
   repository

4. run relevant units tests (see [Unit testing](#unit-testing))

5. update text files:

	- `TODO.md` with all the stuff that has been done, or new stuff that
	needs to be done

	- the top `README.md`

5. commit

	- for simple commits, a one-line message is acceptable; for commits
	for which changes don't fit into single line, summarize them in the
	first line, and describe the changes in detail in the rest of the
	message (usually in a simple list of changes, possibly taken from
	`TODO.md`)
	
	- optional but useful guidelines:
		
		- don't fit multiple independent changes into one commit


# Coding guidelines
	
- write self-explanatory code whenever possible

- only comment code for two reasons:
	
	- explaining some non-self-explanatory piece of code
	
	- outlining the structure of a big function or class with "header"
	comments


# Unit testing

This library relies on [pytest](https://docs.pytest.org/en/7.2.x/) for
unit testing, configured in `pytest.ini`.

All unit tests are located in the `tests` directory, which is also
configured as pytest's `testpaths`, meaning that all tests can be run by
simply invoking `pytest` with no arguments; see pytest's documentation
for details on how to run specific tests.

A number of custom pytest
[markers](https://docs.pytest.org/en/7.2.x/how-to/mark.html) are defined
in `pytest.ini`, to make it convenient to run tests that are scattered
across the `tests` directory, but share some kind of property. These can
be inspected by running:

```
pytest --markers
```

For example, to only run tests marked with `slow`, do:

```
pytest -m slow
```

Instead, to only run tests that are *not* marked with `slow`, do:

```
pytest -m "not slow"
```

## Adding or modifying unit tests

Things to keep in mind when adding or modifying unit tests:

- remember to keep marker decorators updated (e.g. `slow`, `cuda`,
  etc.): always add all those that apply, and remove those that don't
  apply anymore

- try to mimic the structure of the `cgpv` package when adding new
  modules or directories to `tests`


# Writing documentation

Documentation is generated from
[Sphinx](https://www.sphinx-doc.org/en/master/#) sources, which are in
the `docs/source` directory. The `docs/requirements.txt` contains the
packages needed to generate the documentation, assuming the library is
installed. See the [relevant `README.md`
section](README.md#documentation) for instructions on how to build the
documentation.


# Codebase structure

TODO
