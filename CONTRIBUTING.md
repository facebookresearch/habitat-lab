# Contributing to habitat-lab
We want to make contributing to this project as easy and transparent as
possible.



## Notes on Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").
7. We have adopted squash-and-merge as the policy for incorporating PRs into the `main` branch.
8. We would rather have several smaller pull requests that a single large Pull Request since smaller self-contained changes are easier to review.

We accept these types of Pull Requests. Make sure your Pull Requests are not a mix of different types since they have different review processes.

**Docs change** which adds to or changes the documentation

**Refactoring** that changes  the code to improve its functionality or performance

**Dependency Upgrade** of one or several dependencies in habitat

**Bug Fix** which are non-breaking change which fixes an issue

**Development Pull Requests** that add new features to the [habitat-lab](/habitat-lab) task and environment codebase. Development Pull Requests must be small, have unit testing, very extensive documentation and examples. These are typically new tasks, environments, sensors, etc... The review process for these Pull Request is longer because these changes will be maintained by our core team of developers, so make sure your changes are easy to understand!

**Experiments Pull Requests** that add new features to the [habitat-baselines](/habitat-baselines/) training codebase. Experiments Pull Requests can be any size, must have smoke/integration tests and be isolated from the rest of the code. Your code additions should not rely on other habitat-baselines code. This is to avoid dependencies between different parts of habitat-baselines. You must also include a README that will document how to use your new feature or trainer. **You** will be the maintainer of this code, if the code becomes stale and is not supported often enough, we will eventually remove it.

__Note:__ Pull Requests are not the only way to share your work with habitat. You can extend the functionality of Habitat in your own code base by importing and using the Habitat Sim, Habitat Lab, and Habitat Baselines API. If you have a cool project that uses this API, let us know and we might link to your work in our documentation! We encourage using the Habitat-Lab API in your codebase instead of forking the entirety of Habitat-Lab.


## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use [GitHub issues](../../issues) to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Test
We use pytest testing framework and testing data that needs to be downloaded, please make sure that test are passing:
```
python -m pytest
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
- Documentation of PRs is required!

## License
By contributing to habitat-lab, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
