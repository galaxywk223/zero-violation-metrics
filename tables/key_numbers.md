# Key Numbers for Main Text

The table condenses the reported matrix into main-text quantities that can be cited in the main text.

| evaluation question | answer | number | paper role |
| --- | ---: | ---: | ---: |
| Which method preserves reward? | PPO has the highest return. | return=5.376, safe rate=0.331 | Defines the reward-only reference point. |
| Which method has the highest safe rate? | PPOSaute has the highest safe rate but remains below true zero violation. | safe rate=0.827, return retained=21.628% | Separates safe-rate improvement from deployment-level zero violation. |
| Which method is the balanced comparator? | FOCOPS occupies the middle of the return-safety landscape. | return=3.063, safe rate=0.653, unsafe reduction=48.173% | Defines the comparator for future zero-violation-oriented methods. |
| Does mean cost determine zero-violation behavior? | Mean cost and safe rate are strongly related but do not determine persistence. | mean-cost/safe-rate r=-0.89, safe-rate/max-run r=-0.08 | Motivates reporting a metric panel rather than a single safety scalar. |
