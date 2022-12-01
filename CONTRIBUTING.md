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

1. write code (see the [coding guidelines](#coding-guidelines) below)

	- annotate types of everything, unless Python typing issues would
	make the code less readable
	
	- when modifying or extending the API:
	
		- add or modify unit tests (see [unit testing](#unit-testing))
	
		- document API modifications in (see the section about [writing
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

4. run relevant units tests (see [unit testing](#unit-testing))

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

Unit tests for this library use
[`unittest`](https://docs.python.org/3/library/unittest.html). To run
specific unit tests, use:

	`python -m unittest <modified module>`

(TODO how to run all tests?)

See the `unittest` documentation for more details.


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
