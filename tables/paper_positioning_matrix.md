# Paper Positioning Matrix

The table converts related-work families into paper-positioning decisions. It clarifies why the study is a reporting-protocol contribution rather than a new optimizer, a safety-filter method, or a formal guarantee paper.

| paper family | examples | typical evidence | remaining question | Study response |
| --- | ---: | ---: | ---: | ---: |
| CMDP optimizers | CPO; PPO-Lag; PID Lagrangian; FOCOPS; CVPO; Saute RL | Return-cost curves, final return, expected cost, constraint satisfaction | Whether low average cost also means episode-level zero violation | Report safe rate, nonzero frequency, tail severity, and persistence beside return and mean cost. |
| Intervention and hard-safety methods | Shielding; safety layers; recovery policies; CBF filters; verified Safe RL | Safe action sets, intervention frequency, reachability, formal guarantees | How benchmark baselines look before adding direct intervention mechanisms | Keep the baseline matrix as a reference map and avoid claiming formal safety guarantees. |
| Risk and chance-constraint methods | Percentile risk; coherent risk; distributional risk; WCSAC; adaptive chance safeguards | Tail risk, violation probability bounds, worst-case or distributional objectives | Which empirical safety summaries should appear in a benchmark table | Use separate columns for expected cost, event frequency, tail magnitude, and consecutive-run behavior. |
| RL benchmark and reliability papers | Safety Gymnasium; OmniSafe; GUARD; D4RL; RL Unplugged; rliable | Task coverage, fixed protocols, seeds, uncertainty summaries, reproducibility material | What the evaluation protocol can and cannot support | Make claim boundaries explicit and keep the contribution at the reporting-protocol layer. |
