from symplectic_ncp.experts.demonstration import Demonstration, load_demonstrations, save_demonstrations

# The NMPC expert pulls in CasADi/do-mpc; load it lazily so that consumers of
# saved demonstrations do not pay for (or require) the solver stack.
_LAZY = {
    "NMPCExpert": "symplectic_ncp.experts.nmpc",
    "casadi_hamiltonian": "symplectic_ncp.experts.nmpc",
    "rollout_expert": "symplectic_ncp.experts.generate",
    "generate_demonstrations": "symplectic_ncp.experts.generate",
    "demonstration_summary": "symplectic_ncp.experts.generate",
    "demonstrations_path": "symplectic_ncp.experts.generate",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Demonstration", "load_demonstrations", "save_demonstrations", *_LAZY]
