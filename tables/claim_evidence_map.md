# Claim-Evidence Map

| claim | evidence | boundary |
| --- | ---: | ---: |
| Expected-cost safety and episode-level zero-violation safety are not equivalent. | Metric correlations separate mean cost, safe rate, p95 cost, and maximum consecutive cost run. | The evidence is empirical and benchmark-scoped, not a theoretical impossibility result. |
| Mature benchmark baselines do not reach true zero-violation behavior in the reported matrix. | The highest method-level safe rate remains below one, and every method retains nonzero-cost episodes. | The evidence does not cover every Safe RL algorithm, environment, or hyperparameter setting. |
| Reward preservation and episode-level safety form a visible trade-off. | PPO has the highest return and lowest safe rate, while CPPOPID and PPOSaute have higher safe rates and lower returns. | The evidence supports a baseline trade-off map, not a universal Pareto frontier. |
| FOCOPS is the most relevant balanced comparator for future zero-violation-oriented methods. | FOCOPS occupies the middle region across return, mean cost, safe rate, tail cost, and run-length metrics. | FOCOPS is not a zero-violation solution in this matrix. |
| Safe RL evaluation should report a metric panel rather than only mean cost. | Environment slices, seed variability, bootstrap intervals, and rank profiles change across metrics. | The recommendation concerns evaluation reporting; it does not prescribe a new optimization objective. |
