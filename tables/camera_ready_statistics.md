# Camera-Ready Statistics

## Environment-Stratified Bootstrap

Within each method, resample the three seeds separately inside each environment, then equally average the three environment means.

| Method | Metric | Mean | 95% CI |
| --- | --- | ---: | ---: |
| PPO | return | 5.376 | [5.170, 5.582] |
| PPO | safe_rate | 0.331 | [0.298, 0.360] |
| PPO | p95_cost | 46.389 | [44.189, 48.589] |
| PPOLag | return | 2.529 | [2.246, 2.821] |
| PPOLag | safe_rate | 0.700 | [0.662, 0.736] |
| PPOLag | p95_cost | 33.750 | [31.472, 36.067] |
| FOCOPS | return | 3.063 | [2.676, 3.428] |
| FOCOPS | safe_rate | 0.653 | [0.602, 0.702] |
| FOCOPS | p95_cost | 25.572 | [19.756, 31.100] |
| CPO | return | 3.682 | [3.439, 3.914] |
| CPO | safe_rate | 0.620 | [0.580, 0.653] |
| CPO | p95_cost | 31.467 | [28.717, 34.111] |
| CPPOPID | return | 1.102 | [0.769, 1.413] |
| CPPOPID | safe_rate | 0.820 | [0.787, 0.849] |
| CPPOPID | p95_cost | 23.639 | [14.106, 34.344] |
| PPOSaute | return | 1.163 | [1.049, 1.289] |
| PPOSaute | safe_rate | 0.827 | [0.787, 0.869] |
| PPOSaute | p95_cost | 30.044 | [14.250, 51.789] |

## Environment-Sliced Correlations

| Environment | n | Pair | Pearson r |
| --- | ---: | --- | ---: |
| SafetyPointGoal1-v0 | 18 | mean_cost vs. safe_rate | -0.899 |
| SafetyPointGoal1-v0 | 18 | safe_rate vs. max_consecutive_cost_run | 0.330 |
| SafetyPointButton1-v0 | 18 | mean_cost vs. safe_rate | -0.933 |
| SafetyPointButton1-v0 | 18 | safe_rate vs. max_consecutive_cost_run | -0.467 |
| SafetyCarGoal1-v0 | 18 | mean_cost vs. safe_rate | -0.802 |
| SafetyCarGoal1-v0 | 18 | safe_rate vs. max_consecutive_cost_run | 0.285 |

## Method-Sliced Correlations

| Method | n | Pair | Pearson r |
| --- | ---: | --- | ---: |
| PPO | 9 | mean_cost vs. safe_rate | -0.935 |
| PPO | 9 | safe_rate vs. max_consecutive_cost_run | -0.917 |
| PPOLag | 9 | mean_cost vs. safe_rate | -0.908 |
| PPOLag | 9 | safe_rate vs. max_consecutive_cost_run | -0.444 |
| FOCOPS | 9 | mean_cost vs. safe_rate | -0.960 |
| FOCOPS | 9 | safe_rate vs. max_consecutive_cost_run | -0.861 |
| CPO | 9 | mean_cost vs. safe_rate | -0.857 |
| CPO | 9 | safe_rate vs. max_consecutive_cost_run | -0.466 |
| CPPOPID | 9 | mean_cost vs. safe_rate | -0.516 |
| CPPOPID | 9 | safe_rate vs. max_consecutive_cost_run | 0.292 |
| PPOSaute | 9 | mean_cost vs. safe_rate | -0.563 |
| PPOSaute | 9 | safe_rate vs. max_consecutive_cost_run | -0.262 |

## Pairwise Ranking Disagreements

- **PPOLag vs. CPO:** PPOLag has the higher safe rate, whereas CPO has the lower mean cost and higher return.
- **CPPOPID vs. PPOSaute:** PPOSaute has the slightly higher safe rate, whereas CPPOPID has the lower mean and p95 cost.

All statistics are descriptive. The environment slices contain 18 method-seed cells and the method slices contain nine environment-seed cells.
