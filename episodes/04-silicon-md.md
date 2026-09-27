---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Simulate silicon with MACE

We start with a cubic diamond-Si cell repeated twice in each direction:
64 atoms. The pinned MACE potential predicts energy and forces. This run
uses NVE velocity Verlet, a 0.1 fs step, ten warmup steps, and 200 timed
steps. Initial velocities correspond to 300 K but NVE has no thermostat;
the temperature is not held at 300 K.

The ALCHEMI example constructs `AtomicData` for each system, assigns seeded
velocities, and collects the systems into a `Batch`. With one replica the
batch still contains just one physical simulation:

```{literalinclude} ../examples/alchemi_si.py
:language: python
:start-at: def make_batch
:end-before: def run_steps
```

The integrator and neighbor-list hook are explicit. `--integrator nve`
selects `NVE`; `--integrator langevin` selects an NVT Langevin method with
temperature and friction set in the code. Those methods answer different
physical questions and should not share one performance or trajectory claim.

```{literalinclude} ../examples/alchemi_si.py
:language: python
:start-at: def run_steps
:end-before: def main
```

With the Arrhenius environment selected and one GPU allocated, this cell
runs the single trajectory. The helper script mounts the model and this
lesson's Python file read-only into the SIF, so it runs the code shown above.

```{code-cell} ipython3
import json
import subprocess
from pathlib import Path
from IPython.display import Markdown, display

lesson = Path.cwd().parent if Path.cwd().name == "episodes" else Path.cwd()
completed = subprocess.run(
    ["bash", str(lesson / "scripts/run-alchemi.sh"), "--replicas", "1", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
one = json.loads(completed.stdout.strip().splitlines()[-1])
display(Markdown(
    "| Atoms | Timed steps | MD time (s) | Steps/s | Peak Torch memory (GiB) |\n"
    "| ---: | ---: | ---: | ---: | ---: |\n"
    f"| {one['atoms_per_replica']} | {one['measured_steps']} | "
    f"{one['measured_seconds']:.3f} | "
    f"{one['replica_steps_per_second']:.2f} | "
    f"{one['peak_torch_allocated_bytes']/2**30:.2f} |"
))
```

:::{note}
This confirms that the MD run completes and gives one short timing. It does not validate
the model against reference science. PyTorch's peak allocation is not total
GPU memory use.
:::
