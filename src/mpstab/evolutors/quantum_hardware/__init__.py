# Techniques for resynthesizing dressed rotations into circuits runnable on
# real hardware. The optional rustiq dependency is isolated with lazy imports
# inside each module -- importing this package itself never requires it.
from .rustiq_synthesis import (
    build_head_and_residual,
    fold_observable,
    head_counts_only,
    head_to_qibo_circuit,
)
