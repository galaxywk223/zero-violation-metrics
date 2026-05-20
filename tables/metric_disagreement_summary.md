# Metric Disagreement Summary

The table records which method is best and worst under each metric. The point is not a universal ranking, but the fact that the ranking target changes with the metric.

| metric | best method | best value | worst method | worst value | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: |
| return | PPO | 5.376 | CPPOPID | 1.102 | Higher value is better. |
| mean cost | CPPOPID | 4.144 | PPO | 15.513 | Lower value is better. |
| safe rate | PPOSaute | 0.827 | PPO | 0.331 | Higher value is better. |
| nonzero-cost frequency | PPOSaute | 0.173 | PPO | 0.669 | Lower value is better. |
| p95 cost | CPPOPID | 23.639 | PPO | 46.389 | Lower value is better. |
| max consecutive cost run | FOCOPS | 34.333 | CPPOPID | 60.333 | Lower value is better. |
