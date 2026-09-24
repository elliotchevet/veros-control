# CFL and salinity diagnosis — checkpoint 24, Slurm 54631

The 27-step forward replay reproduces training update 25's raw losses exactly as printed in the earlier replay: temperature 1.2117028426151784, salinity 11.17202603328337. dt = 3600 seconds. All changes and diagnostic outputs are inside veros_control; production code, nested checkpointing, and the original checkpoint were not modified.

## Local failure
Two interior wet cells become negative at step 27: (i,j,k)=(2,90,37) and (2,91,37), at 90.5 E, 8.5/9.5 N, depth 29.35 m. The earlier four-cell count included cyclic halo copies.

At (2,90,37), salinity changes from 4.670522445 to -1.991075193 g/kg in the final hour. The applied NN salinity source alone contributes -5.609278574 g/kg. The instantaneous advection tendency times dt is -1.060459091 g/kg, and vertical mixing contributes +0.026263323 g/kg. The instantaneous advection number is not the complete Adams-Bashforth increment; other mixing terms also apply. The source alone exceeds the available salinity.

At (2,91,37), salinity changes from 4.892080252 to -0.739204656 g/kg; the NN source contributes -5.527814115 g/kg.

## CFL measurements
CFL is measured from velocities entering each step; salinity and invalid-field counts are measured after the step. Directional formulas match Veros CFLMonitor. Interior maximum CFL over the replay: x 0.355555, y 0.399266, z 0.766409. No measured directional component exceeds 1.

At the worst salinity cell entering step 27, maxima over bounding faces are x 0.282424, y 0.368898, z 0.179498; their sum is 0.830820. The global maximum face-component sum is 1.092712 at a different cell. This sum is a diagnostic indicator, not a verified exact stability criterion for the full multidimensional scheme. Directional CFL below 1 does not prove stability or positivity of the full source-coupled model.

## Controlled forward comparisons
| NN channels applied | Final minimum interior ocean salinity (g/kg) |
| --- | ---: |
| All, unchanged | -1.991075 |
| Momentum scaled by 0.1 | 23.011651 |
| Momentum disabled | 23.000012 |
| Salinity correction disabled | 14.812943 |
| All NN corrections disabled | 23.090746 |

These runs use identical checkpoint parameters, initial state, forcing, and rollout length. They are ablations, not retraining. The momentum corrections affect the trajectory and hence the tracer NN inputs; the direct NN salinity sink is the dominant local driver of the final negative value. This contradicts a simple explanation based solely on directional CFL exceeding 1. Reduced momentum is promising over this replay but is not a guarantee of longer-run or subsequent-training stability.

Full data: metadata.json names each column and probe location; all.npz and the other variant NPZs contain per-step summaries, local probe histories, and loss terms; results.json contains comparison summaries.
