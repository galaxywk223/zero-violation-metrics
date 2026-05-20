# Claim Boundary

The table translates the empirical result into claim-scoped boundaries. Each row pairs a paper-safe claim with the evidence, the admissible scope, and a common unsupported reading.

| claim | evidence | boundary | not supported |
| --- | ---: | ---: | ---: |
| Expected-cost and zero-violation metrics diverge. | Return/safe-rate correlation is -0.50; mean-cost/safe-rate correlation is -0.89; p95/run-length correlation is 0.61. | Benchmark-scoped empirical evidence; not a theoretical separation theorem. | Expected cost is useless or should be removed. |
| No evaluated mature baseline reaches true zero violation. | The best method-level safe rate is 0.827, below 1.000. | Six methods, three environments, three seeds, and one training budget. | All Safe RL methods fail under all settings. |
| PPO is reward-strong and safety-weak. | PPO return is 5.376; safe rate is 0.331; nonzero-cost frequency is 0.669. | PPO is a reward reference point, not a safety baseline. | High return is acceptable when violation probability remains high. |
| CPPOPID and PPOSaute improve safe rate with large return loss. | PPOSaute safe rate is 0.827 with return 1.163; CPPOPID and PPOSaute retain 20.496% and 21.628% of PPO return. | The result identifies a trade-off, not a cost-free safety improvement. | The safest safe-rate method is automatically the best method. |
| FOCOPS is the balanced comparator. | FOCOPS return is 3.063; safe rate is 0.653; unsafe episodes are reduced by 48.173% relative to PPO. | FOCOPS remains below true zero violation. | FOCOPS solves the zero-violation problem. |
| A metric panel is the paper contribution. | Mean cost, safe rate, tail cost, run length, seed variability, and ranks expose different behavior. | The contribution is an evaluation protocol and evidence map, not a new algorithmic guarantee. | The study reports a successful prototype zero-violation algorithm. |
