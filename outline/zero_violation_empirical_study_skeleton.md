# Zero-Violation Metrics Empirical Study Skeleton

## Working Title

Beyond Expected Cost: An Empirical Study of Episode-Level Zero-Violation Metrics in Safe Reinforcement Learning

## Abstract Claim

Expected cumulative cost is not sufficient to characterize episode-level safety. Mature Safe RL baselines reduce safety costs in different ways, but mean cost, zero-violation probability, tail severity, and violation run length can diverge.

## Introduction

- Safe RL typically reports return and expected cumulative cost.
- Deployment safety often depends on whether an episode contains any violation.
- Episode-level zero-violation metrics expose a different safety axis.
- The study evaluates mature baselines rather than proposing a primary new algorithm.

## Metrics

- Return.
- Mean cost and violation rate.
- Safe rate and nonzero-cost episode frequency.
- p90, p95, and max episode cost.
- Conditional unsafe severity.
- Max consecutive cost run.

## Experimental Setup

- Algorithms: PPO, PPOLag, FOCOPS, CPO, CPPOPID, PPOSaute.
- Environments: SafetyPointGoal1-v0, SafetyPointButton1-v0, SafetyCarGoal1-v0.
- Seeds: 1, 2, 3.
- Training budget: 5,000,000 steps per run.
- Evaluation: 50 episodes, 200 max steps.

## Results

- PPO provides the reward baseline and illustrates unsafe high-return behavior.
- CPPOPID and PPOSaute provide the strongest safe-rate baselines with weak return.
- FOCOPS provides the strongest balanced baseline.
- No mature baseline reaches actual zero-violation behavior.
- Mean cost, safe rate, tail severity, and run length show non-identical ranking behavior.
- Environment-level trade-off facets and method-profile heatmaps summarize where the ranking changes.
- Metric-family, quadrant, and environment-profile figures turn the aggregate matrix into paper evidence rather than an execution trace.
- Three-axis bubble, zero-violation-gap heatmap, safety-signature, and claim-flow figures strengthen the paper-facing evidence chain.
- Statistical-reporting checklist and ladder artifacts connect the evidence matrix to RL evaluation guidance on uncertainty, seed sensitivity, and scoped statistical interpretation.
- Reporting-protocol upgrade, environment-case-study, environment-method scorecard, and protocol-coverage artifacts convert the matrix into paper evidence.

## Discussion

- Expected-cost constraints and episode-level event constraints target different objects.
- A low mean cost can coexist with nonzero episode violation probability.
- A high safe rate can coexist with severe tail failures.
- Zero-violation metrics should be reported alongside expected cost in Safe RL evaluations.
- Literature positioning separates expected-cost optimization, intervention mechanisms, risk objectives, implementation infrastructure, verified safety, and benchmark protocols from the paper's evaluation-layer contribution.
- The literature-to-metric coverage figure makes the reporting gap explicit by mapping adjacent paper families to expected-cost, event-frequency, tail, persistence, and protocol-reliability dimensions.
- Protocol-upgrade artifacts explain how the paper extends return-cost reporting into episode-event, tail, persistence, and claim-boundary reporting layers.
- Statistical comparison should remain descriptive and scope-aware because the matrix has three seeds per environment-method cell rather than enough independent runs for broad dominance claims.
- Reproducibility material should be presented as supporting material for aggregate evidence inspection, not as the paper's primary narrative.

## Limitations

- The study covers six algorithms, three environments, and three seeds.
- The result does not cover all Safe RL methods.
- The result does not establish a new method.
- WCSAC and external tail-risk implementations require separate integration if included.

## Reproducibility

- The study uses a reported 54-cell evaluation matrix.
- Derived tables and figures are generated from aggregate evaluation metrics.
- Reproducibility material is sufficient to reproduce the reported aggregate tables and figures.
- Prototype methods and exploratory variants remain outside the main result boundary.
