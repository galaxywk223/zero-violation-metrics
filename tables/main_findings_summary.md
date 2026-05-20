# Main Findings Summary

The table condenses the reported baseline matrix into four main findings. Each finding is paired with quantitative evidence and its role in the paper narrative.

| id | finding | evidence | paper use |
| --- | ---: | ---: | ---: |
| F1 | Reward-first optimization leaves frequent unsafe episodes. | PPO has the highest return (5.376) but safe rate 0.331 and nonzero-cost frequency 0.669. | Defines the reward reference point and motivates event-level safety reporting. |
| F2 | Safe-rate leaders are conservative and still not zero-violation. | CPPOPID and PPOSaute safe rates are 0.820 and 0.827; return retained versus PPO is 20.496% and 21.628%. | Separates safety improvement from cost-free deployment readiness. |
| F3 | FOCOPS is the balanced comparator, not the zero-violation solution. | FOCOPS return is 3.063, safe rate is 0.653, unsafe episodes reduced versus PPO by 48.173%. | Sets the comparator that future zero-violation-oriented methods must beat. |
| F4 | Safety metrics are coupled but not interchangeable. | Mean-cost/safe-rate correlation is -0.89; safe-rate/max-run correlation is -0.08; p95/max-run correlation is 0.61. | Supports the metric-panel recommendation instead of a single safety scalar. |
