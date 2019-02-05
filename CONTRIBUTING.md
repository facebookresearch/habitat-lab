# Contributing to habitat-api
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

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
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
We use black to format our code, you can use the following commands to format your code prior to submission:

```
pip install black
black habitat test --line-length 79
```

## License
By contributing to habitat-api, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

