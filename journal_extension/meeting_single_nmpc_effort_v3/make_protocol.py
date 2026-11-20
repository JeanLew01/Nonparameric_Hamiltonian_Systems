"""Write protocol.json after expert.py and extract.py have run: input hashes and the evaluation groups."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=['reference_single/benchmark.py','reference_single/initial_states.npz','event_detection/ball_events.py',
        'reference_code_7974f19e/symplectic_ncp/chain/construction.py','reference_code_7974f19e/symplectic_ncp/experts/generate.py',
        'meeting_single_nmpc_effort_v3/expert.py','meeting_single_nmpc_effort_v3/extract.py','meeting_single_nmpc_effort_v3/demonstrations.npz','meeting_single_nmpc_effort_v3/original_library.npz']
groups=[dict(M=M,K=K) for M in range(1,6) for K in (1,8)]+[dict(M=3,K=2),dict(M=3,K=4)]
protocol=dict(study='Effort-minimal NMPC demonstrations, one push per period (one-sided torque in the libration demonstrations), demonstrations 4 and 5 from rest at +-1.5 rad, Algorithm 1 supports with V, transport of whole pushes by fiber realization; 18 Nov 2026',
    frozen_input_sha256={name:sha(ROOT.parent/name) for name in inputs},
    unchanged=['plant, target, input bound |u|<=20 of the plant','500 initial states (seed 2026), 2000 s horizon','placements z=Phi_{jT/K}(x_i), K in {1,2,4,8}',
        'selector, terminal rule, guard, event detection and integration: byte-identical copies of ../meeting_single_nmpc_fiber (policy.py, evaluate.py, fast_access.py, histories.py)'],
    evaluation_groups=groups,horizon=2000.0,summary_horizons=[5.0,10.0,20.0,50.0,100.0,2000.0],
    refinement=dict(groups=[dict(M=M,K=K) for M in (1,2,3,4,5) for K in (1,8)],subdivisions=8,initial_indices=list(range(0,500,10))),
    design_note='Third version, after the user remark of 18 Nov 2026 on Figure 4(b): for K = 1, 2, 4 the success curves shared one slow tail (libration states gained only at K = 8). Two changes against ../meeting_single_nmpc_effort_v2: (1) the libration demonstrations use torque of one sign, so every demonstration pushes once per period; (2) a transported assignment replays the remainder of the push that contains its anchor. Expert bound, eta, start states, Algorithm 1, selector, guard and evaluation are unchanged.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('protocol written')
