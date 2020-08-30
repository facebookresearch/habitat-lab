# Contributing to habitat-lab
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").
7. We have adopted squash-and-merge as the policy for incorporating PRs into the master branch.  We encourage more smaller/focused PRs rather than big PRs with many independent changes.  This also enables faster development by merging PRs into master quickly and reducing the need to rebase due to changes on master.


## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Versioning / release workflow
We use [semantic versioning](https://semver.org/). To prepare a release:
1. Update version numbers.
2. Update the change log.
3. Make sure all tests are passing.
4. Create a release tag with change log summary using the github release interface (release tag should follow semantic versioning as described above)

Stable versions are regularly assigned by Habitat core team after rigorous testing.

## Issues
We use [GitHub issues](../../issues) to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Test
We use pytest testing framework and testing data that needs to be downloaded, please make sure that test are passing:
```
pytest
```

## Check typing
We use mypy to check Python typing and guard API consistency, please make sure next command doesn't complain prior to submission:
```
mypy . --ignore-missing-imports
```

## Coding Style
  - We follow PEP8 and use [typing](https://docs.python.org/3/library/typing.html).
  - Use `black` for style enforcement and linting. Install black through `pip install black`.

  We also use pre-commit hooks to ensure linting and style enforcement. Install the pre-commit hooks with `pip install pre-commit && pre-commit install`.

## Documentation
- Our documentation style is based on Magnum / Corrade and uses [a similar build system](https://mcss.mosra.cz/documentation/doxygen/).
- Documentation of PRs is highly encouraged!

## License
By contributing to habitat-lab, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
