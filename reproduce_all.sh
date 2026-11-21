#!/usr/bin/env bash
# Reproduces every table and figure of the paper (roughly one to two days on a laptop CPU).
# Each step is resumable; rerun this script after an interruption.
# On macOS, prefix with "caffeinate -i" to keep the machine awake.
set -euo pipefail
bash get_data.sh
python run.py --preset paper   2>&1 | tee -a run_paper.log
python run.py --preset revised 2>&1 | tee -a run_revised.log
python report.py --B runs/paper --C runs/revised --out report
python experimental.py --out runs/experiment 2>&1 | tee -a experimental.log
python fig_experiment_paper.py --run runs/experiment --out report
python revision.py  2>&1 | tee -a revision.log
python revision2.py 2>&1 | tee -a revision2.log
python revision3.py 2>&1 | tee -a revision3.log
echo "done: see report/, report_rev/, report_rev2/, report_rev3/"
