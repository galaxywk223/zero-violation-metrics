# Environment Case Studies

The table converts environment slices into case studies. The purpose is to explain why environment-specific reporting is necessary instead of treating the aggregate average as the whole result.

| environment | return leader | safe-rate leader | mean-cost leader | tail leader | FOCOPS profile | best zero-violation gap | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SafetyPointGoal1-v0 | PPO (5.069) | PPOSaute (0.873) | PPOSaute (2.633) | PPOSaute (17.633) | return=3.271; safe rate=0.733; p95=21.500; max run=28.000 | 0.127 | The easiest slice still separates reward leadership from safe-rate leadership; zero-violation reporting remains necessary even when costs are lower. |
| SafetyPointButton1-v0 | PPO (5.082) | CPPOPID (0.760) | CPPOPID (3.640) | CPPOPID (20.617) | return=1.446; safe rate=0.467; p95=34.933; max run=55.000 | 0.240 | The hardest slice exposes the central trade-off: PPO keeps return while safety-oriented methods raise safe rate with severe return loss. |
| SafetyCarGoal1-v0 | PPO (5.979) | CPPOPID (0.873) | CPPOPID (3.573) | CPPOPID (19.950) | return=4.473; safe rate=0.760; p95=20.283; max run=20.000 | 0.127 | The car-control slice highlights tail and persistence behavior; high safe rate does not eliminate the need to inspect p95 cost and maximum run length. |
