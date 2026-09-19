# Project: FL-IENN (undergraduate thesis, IoMT security framework)

Source of truth: `docs/FROZEN_ARCHITECTURE.md`. Read it fully before any task.
Existing reference code: `legacy/Session_Layer_Combined_Diabetes_CVD.ipynb` (do NOT edit it) and `fl_ienn/optim/gmco.py`.

## Rules
1. Work on ONE step at a time, exactly as named in my task prompt. Do not implement later steps or add unrequested features.
2. Never change a decision listed under "FROZEN DECISIONS" or "NO LONGER ALLOWED TO CHANGE". If a task seems to conflict with the frozen doc, or you find a flaw, STOP and report it. Do not silently work around it.
3. Keep compatibility: existing classes (`ECCAuthenticator`, `ExponentialKAnonymity`, `run_session_layer`, `GMCO`) are moved/reused verbatim or extended by subclass / wrapper / optional argument. Never rewrite working code.
4. Variable naming families are mandatory: `gt_*` (ground truth, evaluation only), `eka_prior_*` (EKA only), `pred_*` (IENN output), `enc_*` (encryption routing). Never let `gt_*`, `eka_*`, `pred_*` reach a model input.
5. Never invent or hard-code results. Every number in a report must come from code you actually ran.
6. Every step writes outputs to `reports/<layer>/` (`metrics.json`, `table_*.csv`, `fig_*.png`) and has pytest tests for its "Done when" condition.
7. Reproducibility: all seeds and paths come from `config.yaml`. Use `pathlib`, no hard-coded absolute paths.
8. Style: Python 3.10+, type hints, short docstrings, small functions. Dependencies go in `requirements.txt`.
9. ML training steps must also run on CPU (small smoke-test mode) so they can be tested locally.
10. Do not commit data, artifacts or keys (`.gitignore`).

## Package layout
fl_ienn/{common,session,privacy_security,optim,storage,analysis,app}, notebooks/, data/{raw,processed}, artifacts/, reports/, tests/, docs/, legacy/.

## How to finish every task
Reply with: (a) files created/changed, (b) commands you ran and whether tests passed, (c) the step's "Done when" status, (d) open questions or deviations. Then stop and wait.