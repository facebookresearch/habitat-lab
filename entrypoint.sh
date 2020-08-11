#!/bin/bash
set -e

# activate the habitat conda environment
source activate habitat

exec "$@"
