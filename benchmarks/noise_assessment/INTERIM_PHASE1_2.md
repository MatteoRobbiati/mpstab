# Noise-channels assessment: interim note (Phases 1–2)

2026-11-25. Code at `analytic-gradient` 88cc212, which contains `noise-channels`.
Scripts: `phase2_noise_mass_audit.py`, `common.py`. Results: `results/phase2_mass_audit.{json,csv,png}`.
Phases 3–6 have not started. Stop condition reached: **M_cone ≳ 1 for the realistic use case.**

## Phase 1: literature

All arXiv IDs given in the task exist and match their titles. A subagent read the abstracts, and the method sections where needed.

| Work | Covers | Overlap with us | What we do that it does not |
|---|---|---|---|
| Compiled CAMPS, 2611.24044 | CAMPS with Clifford reuse; coherent and Pauli noise; QEC logical error rates | **Pauli faults as sign flips of residual rotations, ϑ_j = (−1)^ℓ θ_j: same mechanism** | Kraus/non-Pauli channels, χ coherences, exact strata, observables. No code released |
| QuasiStim 2610.15975; Bennink 1703.00111; Hakkaku 2105.04478 | QPD of non-Pauli channels into stabilizer channels (Clifford-only) | Quasi-probability for non-Pauli noise | Non-Clifford rotations, MPS, coherences as amplitudes instead of negative-weight channels |
| Miller et al. 2504.15128 (+ 2603.18457) | Error-generator propagation, BCH plus Taylor, Clifford circuits | Perturbative in noise, conjugation through Cliffords | Non-Clifford circuits, sampled tail |
| PFSR 2603.14670 | Sparse Pauli-frame pure state; Kraus channels sampled as trajectories | Near-Clifford, non-Pauli noise | Density-matrix coherences, MPS, strata |
| Angrisani et al. 2501.13101; PauliPropagation.jl 2505.21606 | Heisenberg Pauli propagation with arbitrary incoherent local noise, non-unital included | Same target (noisy ⟨O⟩) | Schrödinger-picture MPS; unbiased estimator with error bar |
| Aigner et al. 2604.22588 §V.2 | Channels as Σ α_PQ P\|S⟩⟨S\|Q on stabilizer states | **Closest to the ket/bra-frame idea** | Non-stabilizer (MPS) states, sampling; there the cost is exponential in the number of non-diagonal channels |
| Seddon et al. 2002.06181 | Dyadic \|L⟩⟨R\| stabilizer frame simulator | Off-diagonal ket/bra pairs in a quasi-probability simulator | Applies it to noise χ on MPS |
| Myers et al. 2512.07304 | General channel = (1−γ)id + γ·QPD, **stratified by number of faults k**, target Tr(Oρ) | **Closest to the strata idea** | Exact k=0,1 sum, non-Clifford circuits, light cone |
| Subset sampling (Heußen 2309.12774; Bravyi–Vargo 1308.6270) | Fault-weight strata for logical error rates | Stratification | Expectation values |
| CAMPS trajectories with noise (2609.16160, 2609.19116) | Noisy CAMPS through unravelings | CAMPS plus general noise | No public code found |

**What is new.** Sign flips are not new (2611.24044, and Pauli-frame folklore). Neither is quasi-probability over non-Pauli channels, nor stratification by k (2512.07304; subset sampling). We found no prior work that evaluates χ coherences as transition amplitudes between two Clifford-augmented MPS with different sign patterns on ket and bra. Closest prior art: 2604.22588 and 2002.06181, both restricted to stabilizer states. 2601.05131 (frame representations) was not read in full and must be checked before any novelty claim. The honest framing is: "the Pauli-pair/dyadic expansion moved from stabilizer states to CAMPS, where a frame change costs only angle signs", plus a stratified estimator adapted from subset sampling.

**References [1]–[4] of the note.** All exist and support their sentences. Fix: in (c) the log-depth result is Theorem 3 of [2], so cite [2] (or [2, 3]), not [3, 2]. [4] covers Pauli-diagonal noise only; do not use it for non-unital claims. [1] is now PRX Quantum 7, 032001.

**Public competitor code.** No CAMPS-with-noise implementation was found, so the CAMPS-trajectory comparison in Phase 5 cannot run.

## Phase 2: noise-mass audit

**Model.**
- Circuit: KickedIsing of the mitigation tutorial (θ_h=0.7, θ_z=0.3, θ_J=−1.3).
- Device: T1=100 µs, T2=80 µs; 1q gates 30 ns, 2q gates 150 ns, RZ virtual (0 ns, noiseless); `DepolarizingError` with λ=1e-3 (1q) and 5e-3 (2q).
- Thermal relaxation on every gate qubit for the gate time. Thermal relaxation on idle qubits for the idle part of each moment.
- Static ZZ: ζ=50 kHz. **Placeholder**, not given in the task.
- Readout: p0=0.01, p1=0.03. **Placeholder.** It does not enter M because it is folded into the observable.
- "Depth" means moments (circuit layers), so 1 Trotter step = 8 moments with native CNOT, 15 with CZ = H·CZ·H.
- Strata formulas checked against `noisy_expectation` on n=6: identical to machine precision.

**Main results, `physical` variant, native CNOT:**

| n | moments (steps) | obs | L | L_cone | M | M_cone (coh. part) | W0 / W1 / W≥2 | k=1 patterns / transitions | R_cone |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 8 (1) | ZZ | 78 | 60 | 0.21 | 0.14 (0.04) | 0.90 / 0.13 / 0.010 | 37 / 72 | 48 |
| 10 | 32 (4) | ZZ | 315 | 297 | 0.84 | 0.77 | 0.58 / 0.45 / 0.23 | 791 / 378 | 216 |
| 10 | 64 (8) | ZZ | 631 | 613 | 1.68 | 1.61 | 0.32 / 0.52 / 0.78 | 1795 / 786 | 440 |
| 20 | 8 (1) | Z | 158 | 51 | 0.43 | 0.11 (0.03) | 0.92 / 0.10 / 0.006 | 30 / 58 | 49 |
| 20 | 8 (1) | ZZ | 158 | 59 | 0.43 | 0.13 | 0.91 / 0.12 / 0.008 | 35 / 68 | 55 |
| 20 | 8 (1) | Z⊗20 | 158 | 158 | 0.43 | 0.42 | 0.75 / 0.31 / 0.075 | 106 / 200 | 116 |
| **20** | **32 (4)** | **ZZ** | 635 | 521 | 1.72 | **1.37 (0.40)** | 0.38 / 0.52 / **0.59** | 1459 / 650 | 397 |
| 20 | 32 (4) | Z⊗20 | 635 | 635 | 1.72 | 1.71 | 0.30 / 0.51 / 0.84 | 1699 / 806 | 464 |
| **20** | **64 (8)** | **ZZ** | 1271 | 1157 | 3.43 | **3.08 (0.90)** | 0.11 / 0.35 / **2.00** | 3583 / 1458 | 861 |
| 20 | 64 (8) | Z⊗20 | 1271 | 1271 | 3.43 | 3.42 | 0.09 / 0.31 / 2.34 | 3823 / 1614 | 928 |

- Native CZ (`physical_cz`, n=20, local ZZ): M_cone = 0.27 at 15 moments, 0.92 at 30, 2.27 at 60.
- Quasi-norm in the cone, n=20: 1.13 at 8 moments, 1.65 at 32, 2.74 at 64.
- Compile plus light cone: ≤ 0.2 s and ≤ 130 MiB in every configuration, so negligible.

**Verdict.** For n=20, depth ≥ 30 moments and local observables, M_cone ≈ 0.9–1.4, rising to 2.3–3.1 at depth 60. That is clearly in the "≳ 1" regime, where W≥2 is at least as large as W0 and W1. Below 0.3 only at about 1 Trotter step (8–15 moments). If "depth" meant Trotter steps, M_cone at 30 steps would be about 10: far worse.

**What dominates M_cone** (n=20, 32 moments, ZZ; ΔM_cone when the source is removed):

| Source removed | ΔM_cone |
|---|---|
| T1/T2 during 2q gates | 1.08 |
| Static ZZ | 0.63 (it has no mass of its own; its rotations widen the light cone) |
| 2q depolarizing | 0.57 |
| 1q depolarizing | 0.05 |
| Idle thermal | 0.03 |

- Sources interact through the light cone, so these deltas are not additive.
- Coherences (non-unital part of T1) make up about 30% of M_cone; the twirl part is about 70%.
- **Idle is negligible** in a layer-ordered KickedIsing. It dominates only through a queue artefact (next point).

**Artefact to fix regardless of the paper.** `add_idle` layers the gate queue in order. `KickedIsing` queues gates qubit by qubit (`RX q, RZ q, …` and `CNOT RZ CNOT` per bond), so the moments form a staircase. One step becomes 59 moments instead of 8 at n=20, and M_cone at depth 30 goes from 1.4 to about 10 (`stock_queue` rows). The tutorial circuits hit this. `add_idle` also uses a single duration for every moment, so 0-ns virtual-RZ moments get full idle noise.

**Would a plausible change bring M_cone below about 0.3?** M_cone scales roughly linearly with noise strength and with steps before the cone saturates (about 0.33 per step for a local observable at n=20).

| Change | M_cone, n=20, depth ~30 | Enough? |
|---|---|---|
| Shorter circuits | below 0.3 only at ≤ 1 step | Only at ~1 step: too shallow to be interesting for CDR |
| Smaller observables | weight 1 vs weight 2: 1.32 vs 1.37 | No: the cone saturates after ~5 steps anyway |
| Idle noise only where needed | idle contributes ≤ 0.03 | No effect once the circuit is layer-ordered |
| Drop or twirl static ZZ (tunable couplers) | −0.6 | Helps, but still ≈ 0.8 |
| Noise about 4–5× weaker than these placeholders (2q error ≈ 1e-3, T1 ≈ 400 µs) | ≈ 0.3 at depth 30 | Yes, but requires much better hardware |
| Noise about 10× weaker | ≈ 0.3 at depth 60 | Yes, but requires much better hardware |
| Different application: shallow (≤ 2 steps), Clifford-rich, wide circuits with local observables (light cone ≪ n) | Mass stays ≲ 0.3 even at large n, because the cone bounds L_cone | Yes: the only structural lever. Candidates: CDR training circuits at low depth, error-detection/QEC gadgets, near-Clifford echo or Loschmidt circuits |

## Already wrong or overstated in the internal note

1. **"Error ∝ W≥2 ≈ M²/2".** This only holds for M ≪ 1. Measured W≥2 vs M²/2: 0.59 vs 0.93 (M_cone=1.37) and 2.00 vs 4.75 (M_cone=3.08). Beyond M_cone ≈ 1.4, W≥2 is larger than W0 + W1, so the exact strata no longer carry most of the value.
2. **"Competitive in the regime of current processors with error mitigation".** At the placeholder device parameters, the mitigation circuits at depth ≥ 30 are not in the weak-noise regime.
3. **"Idle is the rule generating most locations".** True only with staircase queues. In layer-ordered circuits idle is a few percent; T1 during 2q gates and 2q depolarizing dominate.
4. **"With γ ∼ 1e-3 per gate and a few hundred locations, Π(1+γ_ℓ) stays close to one".** The quasi-norm is already 1.65 at n=20 and 32 moments, and 2.7 at 64.
5. **Novelty.** Sign flips on residual rotations already appear in 2611.24044, and k-stratified quasi-probability sampling in 2512.07304. Only the two-MPS coherence amplitudes appear new.
6. **Not yet measured.** Phase 3 has not been run, but the k=1 stratum alone already needs about 1.5k MPS patterns plus about 650 full-sweep transitions, each over about 400 rotations. The "exact k=1 is cheap" assumption needs that measurement.

## Decisions needed before Phases 3–6

1. Confirm the placeholders (ZZ, readout) and the definition of "depth": moments, or Trotter steps.
2. Should the KickedIsing queue order / `add_idle` duration issue be fixed in mpstab (a real bug for the tutorials), or only worked around in the benchmarks?
3. Continue with Phases 3–6, or retarget to a shallow / Clifford-rich / weaker-noise setting where M_cone ≲ 0.3?
