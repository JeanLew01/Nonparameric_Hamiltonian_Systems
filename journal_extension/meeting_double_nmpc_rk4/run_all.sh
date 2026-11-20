#!/bin/sh
# Pipeline of the RK4 version of the double-pendulum study (after demonstrations.py).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY verify_demonstrations.py > verify_demonstrations.log
$PY certified_radius.py > certified_radius.log
HCP_WORKERS=12 $PY construct.py reuse > construct_reuse.log
$PY run.py --workers 14 --last 500 > run_500.log
HCP_N=500 $PY summarize.py > summarize_500.log
$PY run_fine.py --workers 14 > run_fine.log
HCP_N=500 HCP_WORKERS=12 $PY check.py > check_500.log
echo PIPELINE_DONE
