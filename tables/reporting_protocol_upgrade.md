# Reporting Protocol Upgrade

The table records the paper's reporting move from conventional return-cost summaries to an episode-level zero-violation metric panel. It is a writing aid for keeping the manuscript in evaluation-paper form.

| reporting layer | reported quantities | answered question | question left open | paper use |
| --- | ---: | ---: | ---: | ---: |
| Conventional return-cost summary | Return; mean episode cost | How much cost is accumulated on average? | How often does any episode contain a violation? | Defines the baseline reporting convention that the study extends. |
| Episode-event reporting | Safe rate; nonzero-cost episode frequency | How often is the episode cost exactly zero? | How severe are the residual unsafe episodes? | Makes zero-violation behavior visible as an empirical target. |
| Tail-severity reporting | p90 cost; p95 cost; max episode cost; conditional unsafe severity | How large are the remaining unsafe episodes? | Are violations isolated or temporally persistent? | Separates rare severe failures from frequent mild violations. |
| Temporal-persistence reporting | Maximum consecutive cost-positive run | Do violations persist across consecutive steps? | Whether a new optimizer or intervention mechanism can remove the residual gap. | Prevents high safe rate from being read as stable violation-free behavior. |
| Claim-boundary reporting | Supported claims; unsupported readings; benchmark scope | Which conclusions are justified by the evidence matrix? | Generalization beyond the evaluated methods, tasks, seeds, and budget. | Keeps the paper in the benchmark/evaluation genre and limits claims to the reported evidence. |
