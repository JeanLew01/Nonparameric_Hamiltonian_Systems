#!/bin/sh
# Full pipeline of the study (about 25 min on 16 cores).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY expert.py
$PY verify_demonstrations.py > /dev/null
$PY extract.py
$PY make_protocol.py
$PY construct.py --workers 12
$PY evaluate.py direct_fiber --mode main --workers 6
$PY evaluate.py direct_fiber --mode refinement --workers 6
$PY report.py
echo PIPELINE_DONE
