# Training memory investigation — 2026-09-21

Baseline job: 54413 (H100, 27 steps, sample every 27 steps, interpolated GLORYS target, balanced loss).
Job stopped intentionally after three completed updates and the cache-clearing measurement. Job 54414 was canceled while queued.

| Measurement | Live arrays | Array GiB |
| --- | ---: | ---: |
| g1_calibration_before | 368 | 5.367 |
| update1_end | 697 | 10.260 |
| update2_end | 1011 | 15.152 |
| update3_end | 1313 | 20.045 |
| update3_after_clear_caches | 1313 | 20.045 |

## Findings

- Each tendency diagnostics call retains approximately 4.89 GiB of arrays from its two temporary simulation states.
- Calling jax.clear_caches() after update 3 did not reduce the live-array count or bytes.
- veros/core/external/solvers/__init__.py memoize uses a strong dictionary key containing each state. A CPU weak-reference test confirms the discarded state stays alive until the solver cache releases it.
- A separate tracer leak was confirmed in the NN step: simulation._nn_params and simulation._nn_model_batched retained traced parameters/epsilon. Both stepping helpers now restore prior NN hooks in a finally block; four CPU regression tests pass.

## Validated contained fix

Veros source was not modified. The weak-key patch was not applied.

The original project now scopes the legacy solver cache around tendency diagnostics and AD calls, releasing newly created temporary-state entries and preserving existing entries. The NN stepping helpers also restore previous simulation NN attributes to prevent tracer escape.

GPU validation job 54417 completed four updates without global JAX cache clearing. At update 3, live-array memory was 5.411 GiB (baseline: 20.045 GiB). Full NN parameters, optimizer state, loss denominators, and training history excluding elapsed time were bitwise identical to baseline job 54413 after three updates. See comparison_54413_54417.json.

A smaller residual remains: about 15 MiB/update in two-dimensional pressure-solver arrays. Diagnostic job 54480 confirmed that clearing the compiled solver wrapper did not eliminate it. Per the user's request, further work on this residual was stopped. The attempted solver-reuse edit did not execute, and is not in production code.

Original source files with the memory fixes:
- /Odyssey/private/e25cheve/control/veros_control/training/ad_benchmark_root.py
- /Odyssey/private/e25cheve/control/veros_control/training/ad_benchmark.py
- /Odyssey/private/e25cheve/control/veros_control/training/train.py

Logs and detailed allocation snapshots remain in this directory. No active memory-test jobs remain.
