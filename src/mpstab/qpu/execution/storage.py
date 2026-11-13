"""
Saving a measurement plan's circuits, so they can run elsewhere and later.

:func:`save_circuits` writes them to a folder::

    out_folder/
        planned_circuits/circ_0000.qasm, circ_0001.qasm, ...
        shots.json

That is all a device backend needs: one OpenQASM 2 file per circuit, and the
``{file name: shots}`` map, in plan order.

The circuits are written once and never rebuilt: rustiq draws the residual's
gates at random, so building the same plan again can give a different circuit.
"""

from __future__ import annotations

import json
from pathlib import Path

from mpstab.qpu.execution.qasm import plan_to_qasm
from mpstab.qpu.measurement.abstract import MeasurementPlan

CIRCUITS_FOLDER = "planned_circuits"
SHOTS_FILE = "shots.json"


def save_circuits(plan: MeasurementPlan, out_folder) -> None:
    """
    Write the circuits and shots of ``plan`` to ``out_folder``, creating it as
    needed.

    File names are zero-padded, so sorting them gives the plan order, which is
    the order :func:`~mpstab.qpu.measurement.estimate` expects the
    counts in.

    Args:
        plan: the plan to save.
        out_folder: the folder to write into.

    Raises:
        FileExistsError: if ``out_folder`` already holds circuits. Replacing
            them would orphan any counts already measured from them.
    """
    out_folder = Path(out_folder)
    for name in (CIRCUITS_FOLDER, SHOTS_FILE):
        if (out_folder / name).exists():
            raise FileExistsError(
                f"{out_folder / name} already exists: {out_folder} holds circuits."
            )

    texts = plan_to_qasm(plan)
    width = max(4, len(str(len(texts) - 1)))
    names = [f"circ_{index:0{width}d}.qasm" for index in range(len(texts))]

    (out_folder / CIRCUITS_FOLDER).mkdir(parents=True)
    for name, text in zip(names, texts):
        (out_folder / CIRCUITS_FOLDER / name).write_text(text)
    shots = {name: int(count) for name, count in zip(names, plan.shots)}
    (out_folder / SHOTS_FILE).write_text(json.dumps(shots, indent=2))
