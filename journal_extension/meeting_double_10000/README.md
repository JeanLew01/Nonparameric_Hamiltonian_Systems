# Files of an earlier double-pendulum study

Only what `../meeting_double_nmpc_rk4/construct.py` reads is kept here:

- `benchmark.py`: the model and numerical kernel. `construct.py` checks that
  it is identical to the copy in `../meeting_double_nmpc_rk4`.
- `up_library.npz`, `down_library.npz`: the frozen libraries of the earlier
  study, read by the ablation variant `frozen_rows` of `construct.py`. The
  libraries used in the paper are in `../meeting_double_nmpc_rk4/reuse`.
