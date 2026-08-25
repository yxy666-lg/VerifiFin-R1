# Data

Only synthetic schema examples are versioned here. Full datasets and generated trajectories are intentionally excluded.

Expected local layout:

```text
data/
  raw/       # legally obtained source datasets
  sft/       # generated SFT splits
  prm/       # generated and labelled trajectories
  grpo/      # GRPO prompts
  eval/      # held-out evaluation files
  examples/  # synthetic public examples
```

Do not commit licensed benchmark questions, API-generated private data, access tokens, or model outputs containing restricted source text.
