# Literature Metric Coverage

The table maps adjacent paper families to the metric families needed for episode-level zero-violation reporting. Coverage values are qualitative writing aids: 0 means mostly absent, 0.25 means indirect, 0.5 means partial, 0.75 means prominent, and 1 means central. The table supports paper positioning and does not rank prior work.

| research line | representative papers | expected cost | episode event | tail severity | temporal persistence | protocol reliability | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expected-cost Safe RL optimizers | CPO; PPO-Lag; PID Lagrangian; FOCOPS; Saute RL; CVPO | 1.000 | 0.250 | 0.250 | 0.000 | 0.500 | Core baselines optimize or report cost constraints, but episode-level event metrics are usually secondary. |
| Safety benchmark substrates | Safety Gym; Safety-Gymnasium; SafeLife; AI Safety Gridworlds | 0.750 | 0.500 | 0.250 | 0.000 | 1.000 | Benchmark papers make safety evaluation concrete, but the reported safety object varies by suite. |
| Implementation infrastructure | OmniSafe; Stable-Baselines3; CleanRL | 0.500 | 0.250 | 0.000 | 0.000 | 1.000 | Infrastructure papers justify controlled implementations and reproducible comparisons. |
| Risk and chance constraints | Coherent risk; chance-constrained safe RL; WCSAC; CVaR-PPO | 0.500 | 0.750 | 1.000 | 0.250 | 0.500 | Risk papers motivate safety summaries beyond the mean, especially probability and tail metrics. |
| Zero or bounded violation methods | Zero-constraint-violation primal-dual; Triple-Q; Safe Set Actor-Critic | 0.500 | 1.000 | 0.500 | 0.250 | 0.500 | Method-level zero-violation papers should be separated from reporting-protocol evidence. |
| RL evaluation reliability | RE-EVALUATE; rliable; seed-sensitivity studies; AdaStop | 0.250 | 0.250 | 0.250 | 0.250 | 1.000 | Evaluation papers motivate intervals, independent runs, and explicit claim boundaries. |
| This empirical study | Six mature baselines; three environments; three seeds | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | The study fills the reporting gap by placing all metric families in one baseline matrix. |
