#!/usr/bin/env bash
# Downloads the open actuator and tensile tests of Gorges et al. (2025),
# Zenodo doi:10.5281/zenodo.15657324 (CC BY 4.0), into experimental/.
set -euo pipefail
mkdir -p experimental
base="https://zenodo.org/records/15657324/files"
for f in ActuatorTest_SMA300um.txt AusteniteTensileTest_SMA300um.txt MartensiteTensileTest_SMA300um.txt; do
  if [ ! -s "experimental/$f" ]; then
    echo "downloading $f"
    curl -L --fail -o "experimental/$f" "$base/$f?download=1"
  fi
done
echo "data in experimental/"
