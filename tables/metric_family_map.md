# Metric Family Map

The table groups the reported metrics by the safety question each family answers. It supports the paper's argument that return, expected cost, violation frequency, tail severity, and temporal persistence should be reported together.

| metric family | metrics | evaluation question | Reported evidence | paper role |
| --- | ---: | ---: | ---: | ---: |
| Task performance | Return | How much task performance remains after safety optimization? | PPO has the highest return; CPPOPID and PPOSaute retain only about one fifth of PPO return. | Prevents treating safer policies as automatically preferable when task performance collapses. |
| Expected-cost safety | Mean cost; violation rate | How much safety cost is accumulated on average? | Mean cost separates PPO from safety-aware baselines but does not determine zero-violation probability. | Connects the study to the standard CMDP reporting convention. |
| Episode event safety | Safe rate; nonzero-cost episode frequency | How often does an episode contain any violation? | All evaluated methods retain nonzero-cost episodes; the best method-level safe rate remains below one. | Defines the paper's central zero-violation reporting object. |
| Tail severity | p90 cost; p95 cost; max episode cost; conditional unsafe severity | How severe are the remaining unsafe episodes? | High safe rate can coexist with larger residual tails, especially across seed/environment slices. | Separates rare-event magnitude from event frequency. |
| Temporal persistence | Maximum consecutive cost run | Are violations isolated contacts or sustained unsafe runs? | Safe-rate and max-run correlation is weak, motivating a separate persistence metric. | Connects the empirical protocol to recent consecutive-violation safe-exploration metrics. |
