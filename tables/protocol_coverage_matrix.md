# Protocol Coverage Matrix

The table states what the reported evidence covers and what it does not cover. It supports submission text and supporting-material documentation.

| coverage axis | reported coverage | evidence object | claim supported | claim boundary |
| --- | ---: | ---: | ---: | ---: |
| Method coverage | 6 mature baselines | PPO, PPOLag, FOCOPS, CPO, CPPOPID, and PPOSaute | Baseline landscape and comparator selection | Not all Safe RL algorithms or external implementations |
| Task coverage | 3 Safety-Gymnasium tasks | PointGoal, PointButton, and CarGoal safety tasks | Environment-sliced trade-off behavior | Not a full robotics, manipulation, or real-world benchmark |
| Run coverage | 54 completed method-environment-seed cells | Every method-environment-seed cell is present | Descriptive aggregate comparison under a fixed protocol | Not enough for broad universal dominance claims |
| Metric coverage | Return, expectation, event frequency, tail, and persistence | Mean cost, safe rate, nonzero frequency, p95 cost, and max run | Expected-cost and zero-violation metrics are not interchangeable | Not a formal proof of metric non-equivalence |
| Artifact coverage | Derived tables, figures, notes, and paper skeleton | Deterministic aggregate-evidence generator | Inspectability of the reported aggregate evidence pack | No primary simulator traces, policy parameter files, or large result directories |
