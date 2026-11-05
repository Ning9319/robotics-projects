# Process noise Q from field data (homework3/estimate_q.py, homework3/replay.py)

## Method
Ground truth for each motion comes from the tags, not from the SLAM map: when the stops before
and after one TURN/DRIVE both see >= 2 of the same tags (spread >= 0.25 m), a 2D rigid fit of
those tags' robot-frame positions gives the robot's actual motion (tag "visual odometry").
Motion error = actual - motion predicted by the EKF from the wheel commands.
Runs: square_test4, square_run2, octagon_run4, square_final, octagon_final
(41 drive segments of 0.30 m, 43 in-place turns of 11-29 deg).

## Measured motion errors
| motion | quantity | mean | std |
|---|---|---|---|
| drive 0.30 m | along-track | -1.41 cm | 0.78 cm |
| drive 0.30 m | lateral | -0.17 cm | 0.61 cm |
| drive 0.30 m | heading | +0.19 deg | 0.51 deg |
| CCW turn (n=21) | extra rotation vs prediction | +2.55 deg | 2.26 deg |
| CW turn (n=22) | extra rotation vs prediction | -4.95 deg | 1.60 deg |
| any turn | position shift of the robot reference point | 1.55 cm (RMS 1.94 cm) | |

Systematic parts: the rover drives ~4.7% shorter than the HW1 speed model; short CCW spins
coast ~2.5 deg; short CW spins turn ~5 deg less than the HW1 model (static friction); the
skid-steer rover does not spin exactly about the reference point (~1.5 cm shift per turn,
not modelled in the online Q).

## Data-driven Q (RMS, bias included; output/q_estimate.json)
| term | data value | online (initial, conservative) value |
|---|---|---|
| var_forward_per_m | 8.6e-4 m^2/m (1.6 cm per 0.3 m) | 1.0e-2 (5.5 cm per 0.3 m) |
| var_lateral_per_m | 1.3e-4 m^2/m (0.6 cm per 0.3 m) | 2.5e-3 (2.7 cm per 0.3 m) |
| var_heading_per_m | 2.9e-4 rad^2/m (0.5 deg per 0.3 m) | 1.9e-2 (4.4 deg per 0.3 m) |
| var_heading_per_rad | 2.4e-2 rad^2/rad | 6.25e-2 + (8 deg)^2 per turn |
| turn position std | 1.4 cm per axis per turn | 0 |

## Consistency check by re-processing the recorded runs (replay.py), 681 landmark updates
| Q / R | NIS mean | NIS median | NIS > 9.21 (gate) |
|---|---|---|---|
| online: initial Q, R x2 | 0.38 | 0.15 | 0 |
| data Q x1, R x1 | 97.8 | 1.10 | 138 |
| data Q x2, R x1 | 0.89 | 0.43 | 0 |
| data Q x3, R x1 | 0.63 | 0.32 | 0 |
| data Q x2, R x2 | 0.75 | 0.29 | 0 |
| chi-square, 2 DoF (ideal) | 2.00 | 1.39 | ~1% |

* With data Q x1 the bulk of the updates is consistent (median 1.10 vs ideal 1.39), but turn
  errors are heavy-tailed (e.g. one 0.25 s pulse did not move the rover at all) and the
  segments used above exclude the hardest turns (those after which < 2 common tags remain).
  One such turn in octagon_final made the filter inconsistent for the rest of the run.
* Data Q x2 is the smallest tested scale with zero gate violations over all 681 updates.
* Map accuracy is insensitive to these choices (final runs: shape error 1.3-1.5 cm in every
  setting); Q mainly sets how large the covariance ellipses are. The online runs used the
  conservative initial Q, so their ellipses are larger than the actual errors.
