"""Shared inputs of the effort-minimal NMPC study (single pendulum, 16 Nov 2026).

Same interface as ../meeting_single_nmpc_fiber/common.py, so that policy.py,
evaluate.py, fast_access.py and histories.py are byte-identical copies of the
files there.  The plant, target, 500 initial states and all numerical routines
come from ../reference_single/benchmark.py.  The original assignments are the
ones written here by extract.py from the demonstrations of expert.py.
"""
from pathlib import Path
import hashlib, json, os, sys
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT
PARENT=ROOT.parent/'reference_single'
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'_cache')
sys.path.insert(0,str(PARENT))
import benchmark as b
import numpy as np

VARIANTS=('direct_fiber',)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_sources():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in protocol['frozen_input_sha256'].items():
        assert sha(ROOT.parent/name)==digest,name
    return protocol

def save_npz(path,**arrays):
    path=Path(path)
    with path.with_suffix('.pending').open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    path.with_suffix('.pending').replace(path)

def library_for(payload,M,K):
    mask=payload['introduced_K']<=K
    fields={'centers','radii','leads','demo_ids','anchor_times','parent','shift','is_transport','angles','values','counts','introduced_K','phase_fraction','period','original_source','endpoint_error','inverse_radius'}
    selected={k:(v[mask] if k in fields else v) for k,v in payload.items()}
    return b.Library(selected,M,True)

def output_times():
    return np.r_[np.arange(0,20.00001,.05),np.arange(20.5,150.00001,.5),np.arange(160.,2000.00001,10.)]
