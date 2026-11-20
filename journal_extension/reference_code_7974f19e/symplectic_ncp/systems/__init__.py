from symplectic_ncp.systems.base import HamiltonianSystem, wrap_angle
from symplectic_ncp.systems.single_pendulum import SinglePendulum
from symplectic_ncp.systems.spring_mass import SpringMass

SYSTEMS = {
    SpringMass.name: SpringMass,
    SinglePendulum.name: SinglePendulum,
}


def make_system(name: str, **kwargs) -> HamiltonianSystem:
    try:
        return SYSTEMS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}") from exc


__all__ = ["HamiltonianSystem", "SinglePendulum", "SpringMass", "SYSTEMS", "make_system", "wrap_angle"]
