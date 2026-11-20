#!/bin/sh
# Redraw the four result figures of Section 5 from the stored runs (nothing is
# recomputed) and write the PDFs and PNG previews into figures/:
#   single_success_by_demonstrations  success against the demonstration budget M (meeting_single_nmpc_effort_v3, horizons 20 s and 120 s)
#   single_reeb_time                  residual, success and first wait against time at M = 3 (meeting_single_nmpc_effort_v3)
#   poincare_diagnostic               section visitation and coverage under exploration (meeting_dynamics_kappa_half_decade)
#   double_policy_time                the four policies on the double pendulum (meeting_double_nmpc_rk4, 500 initial states, pointwise 95% confidence bands)
# Usage: ./sync_figures.sh [output_folder]      set PYTHON to choose the interpreter
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
HCP_SINGLE_STUDY=meeting_single_nmpc_effort_v3 HCP_SINGLE_SHORT=20 HCP_SINGLE_LONG=120 HCP_SINGLE_XMAX=110 \
HCP_DOUBLE_STUDY=meeting_double_nmpc_rk4 HCP_DOUBLE_N=500 HCP_DOUBLE_BAND=ci \
"${PYTHON:-python3}" paper_figures.py 9 "${1:-figures}"
