# Abstaining on weak evidence

Confidence = best similarity x vote share. Target precision 0.95, fixed before looking at test.

Threshold chosen on validation (410 lines): **0.7120**

| set | precision on answered | coverage | answered | precision if always answering |
|---|---|---|---|---|
| validation | 0.953 | 0.944 | 387/410 | 0.912 |
| **test** | **0.971** | **0.866** | 547/632 | 0.888 |

## Test, per category

| category | lines | answered right | answered wrong | abstained |
|---|---|---|---|---|
| filesystem_mount_failure | 50 | 0 | 12 | 38 |
| io_connection_lost | 50 | 45 | 0 | 5 |
| job_launch_failure | 32 | 32 | 0 | 0 |
| kernel_crash | 50 | 49 | 1 | 0 |
| link_hardware_failure | 50 | 49 | 0 | 1 |
| memory_cpu_fault | 50 | 49 | 0 | 1 |
| normal | 300 | 269 | 1 | 30 |
| power_control_failure | 50 | 38 | 2 | 10 |

## Precision vs coverage

| threshold | validation precision | validation coverage | test precision | test coverage |
|---|---|---|---|---|
| 0.00 | 0.912 | 1.000 | 0.888 | 1.000 |
| 0.50 | 0.912 | 1.000 | 0.905 | 0.981 |
| 0.60 | 0.912 | 0.998 | 0.908 | 0.978 |
| 0.70 | 0.928 | 0.980 | 0.932 | 0.934 |
| 0.75 | 0.982 | 0.807 | 0.995 | 0.701 |
| 0.80 | 1.000 | 0.793 | 0.998 | 0.657 |
| 0.85 | 1.000 | 0.776 | 1.000 | 0.603 |
| 0.90 | 1.000 | 0.710 | 1.000 | 0.481 |
| 0.95 | 1.000 | 0.576 | 1.000 | 0.337 |
| 0.99 | 1.000 | 0.502 | 1.000 | 0.331 |
| 1.00 | 1.000 | 0.502 | 1.000 | 0.329 |

Curve: `20261103-time-test-abstain.svg`
