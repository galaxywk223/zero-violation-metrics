# Statistical Reporting Checklist

The checklist turns statistical-comparison guidance into evidence requirements. It frames the aggregate matrix as scoped descriptive evidence rather than a universal significance claim.

| layer | evidence object | evaluation question | paper use |
| --- | ---: | ---: | ---: |
| Protocol scope | 6 methods x 3 environments x 3 seeds | Which population of benchmark claims is covered? | Define the evaluated scope before interpreting rankings. |
| Independent runs | Environment-method-seed cells | Are aggregate scores built from separate executions? | Support descriptive comparisons while avoiding universal dominance claims. |
| Metric families | Return, mean cost, safe rate, tails, max run | Which safety object is being compared? | Prevent expected-cost safety from being treated as zero-violation safety. |
| Uncertainty context | Seed variability and bootstrap intervals | How stable are method-level summaries? | Report uncertainty without turning three seeds into strong significance claims. |
| Claim boundary | Supported and unsupported claim table | Which conclusions are justified by this evidence? | Separate metric-reporting evidence from new-method or impossibility claims. |
