#!/usr/bin/env bash

# Propagate failures properly
set -e

mcss_path=../../habitat-sim/docs/m.css

# Regenerate the compiled CSS file (yes, in the sim repository, to allow fast
# iterations from here as well)
$mcss_path/css/postprocess.py \
  ../../habitat-sim/docs/theme.css \
  $mcss_path/css/m-grid.css \
  $mcss_path/css/m-components.css \
  $mcss_path/css/m-layout.css \
  ../../habitat-sim/docs/pygments-pastie.css \
  $mcss_path/css/pygments-console.css \
  $mcss_path/css/m-documentation.css \
  -o ../../habitat-sim/docs/theme.compiled.css

$mcss_path/documentation/python.py conf-public.py

# The file:// URLs are usually clickable in the terminal, directly opening a
# browser
echo "------------------------------------------------------------------------"
echo "Public docs were successfully generated to the following location. Note"
echo "that the search functionality requires a web server in this case."
echo
echo "file://$(pwd)/../../habitat-sim/build/docs-public/habitat-lab/index.html"
