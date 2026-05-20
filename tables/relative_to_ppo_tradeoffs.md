# Relative-to-PPO Trade-Offs

The table uses PPO as the reward-only reference point. Positive safety percentages indicate reductions relative to PPO's unsafe-episode frequency, mean cost, or p95 cost. Return retention reports the fraction of PPO return preserved by each method.

| method | return | return retained % | return delta | safe rate | safe-rate gain | unsafe episode reduction % | mean cost reduction % | p95 cost reduction % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | 5.376 | 100.000 | 0.000 | 0.331 | 0.000 | 0.000 | 0.000 | 0.000 |
| PPOLag | 2.529 | 47.043 | -2.847 | 0.700 | 0.369 | 55.150 | 51.225 | 27.246 |
| FOCOPS | 3.063 | 56.975 | -2.313 | 0.653 | 0.322 | 48.173 | 62.341 | 44.874 |
| CPO | 3.682 | 68.475 | -1.695 | 0.620 | 0.289 | 43.189 | 55.780 | 32.168 |
| CPPOPID | 1.102 | 20.496 | -4.275 | 0.820 | 0.489 | 73.090 | 73.285 | 49.042 |
| PPOSaute | 1.163 | 21.628 | -4.214 | 0.827 | 0.496 | 74.086 | 69.431 | 35.234 |
