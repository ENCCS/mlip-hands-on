"""Relax structures one at a time with ASE, or all at once with TorchSim."""

import time
from dataclasses import dataclass, field

from ase import Atoms
from ase.optimize import FIRE


@dataclass
class RelaxResult:
    wall_s: float
    energies: list[float]
    relaxed: list[Atoms] = field(default_factory=list)


class SerialRelaxer:
    def __init__(self, make_calc, fmax: float, max_steps: int):
        self.make_calc = make_calc
        self.fmax = fmax
        self.max_steps = max_steps

    def run(self, structures: list[Atoms]) -> RelaxResult:
        calc = self.make_calc()
        energies = []
        start = time.perf_counter()
        for original in structures:
            atoms = original.copy()
            atoms.calc = calc
            FIRE(atoms, logfile=None).run(fmax=self.fmax, steps=self.max_steps)
            energies.append(float(atoms.get_potential_energy()))
        return RelaxResult(time.perf_counter() - start, energies)


class BatchedRelaxer:
    def __init__(self, model, fmax: float, max_steps: int, autobatch: bool = False):
        self.model = model
        self.fmax = fmax
        self.max_steps = max_steps
        self.autobatch = autobatch

    def run(self, structures: list[Atoms]) -> RelaxResult:
        import torch_sim as ts

        start = time.perf_counter()
        state = ts.optimize(
            system=structures,
            model=self.model,
            optimizer=ts.Optimizer.fire,
            convergence_fn=ts.generate_force_convergence_fn(force_tol=self.fmax),
            max_steps=self.max_steps,
            autobatcher=self.autobatch,
            pbar=True,
        )
        wall = time.perf_counter() - start
        energies = [float(e) for e in state.energy]
        return RelaxResult(wall, energies, ts.io.state_to_atoms(state))
