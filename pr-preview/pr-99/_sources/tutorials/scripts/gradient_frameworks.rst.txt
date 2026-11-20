Gradients against PennyLane and qiboml
======================================

The cost of one gradient of the transverse-field Ising energy on a
hardware-efficient ansatz, from 8 to 60 qubits: :meth:`~mpstab.HSMPO.gradient`
against PennyLane (``lightning.qubit`` adjoint, ``default.tensor`` with the
parameter-shift rule) and qiboml (``Adjoint`` and ``PSR`` on a statevector,
``PSR`` on a qibotn MPS). Statevector methods stop at 24 qubits, and the
parameter-shift rule is timed on a subset of the parameters and scaled.

It needs PennyLane, qiboml and qibotn next to mpstab, in a Python 3.12
environment:

.. code-block:: bash

   uv pip install -e . pennylane jax \
       "qiboml @ git+https://github.com/qiboteam/qiboml" \
       "qibotn @ git+https://github.com/qiboteam/qibotn"
   python examples/scripts/gradient/gradient_frameworks.py

.. literalinclude:: ../../../examples/scripts/gradient/gradient_frameworks.py
   :language: python
