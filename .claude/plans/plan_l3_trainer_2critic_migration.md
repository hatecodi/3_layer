# Plan — finish the L3 1-critic → 2-critic migration + rerun E1

## Context

`c3ff7ba` (2026-08-16) changed `L3WorkerTrainer` from one critic to two
(`critic_local` + `critic_global`) and decomposed `L3Transition` into
`reward_local`/`reward_global` with two GAEs. This is a **semantic** change — the
policy gradient differs, so pre-8/16 policies are a different algorithm.

The migration was left half-done. Migrated: `run_train_hierarchical.py`,
`eval_protocol.py`, `analyze_l3_behavior.py`, `compare_sa_l3_behavior.py`.
Everything else still speaks the old API and crashes.

Goal (user-approved scope): finish the call-site migration, then rerun E1
(`bench_l3_latency.py`). Separately: verify convergence-CSV provenance. Do **not**
retrain anything in this plan.

## Why no backwards-compat shim

Making `critic_global` optional / silently reusing the local critic would let these
scripts run with 1-critic semantics while appearing fixed — re-creating exactly the
version-mixing bug we are correcting. Old 1-critic checkpoints therefore become
**unloadable** by these scripts, by design.

## Edits

Per-site: add a second `CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=<same>)`, pass
it as `critic_global`, load it from `ck["l3_critic_global"]`, save it under that key.

| File | Line | Problem |
|---|---|---|
| `scripts/train_l3_convergence.py` | 110, 195 | 2-arg ctor; saves `l3.critic.state_dict()` (attr gone) |
| `scripts/bench_l3_latency.py` | 109, 111 | 2-arg ctor; loads `ck["l3_critic"]` |
| `src/experiment/run_train_l3_only.py` | 95, 210, 225 | 2-arg ctor; saves `l3_critic` |
| `scripts/continue_with_vt.py` | 80, 81, 207, 214 | 2-arg ctor; loads/saves `l3_critic`; `l3_trainer.critic` |
| `scripts/train_vt_staged.py` | 339, 352 | 2-arg ctor; saves `l3_critic` |
| `scripts/validate_l3_schedule.py` | 177, 182 | 2-arg ctor; loads `l3_critic` |
| `scripts/robustness_test.py` | 38-45 | 2-arg ctor; hardcoded 1-critic ckpt; `hidden_dim=64` |
| `scripts/plot_train_diagram.py` | 44, 50, 52 | 2-arg ctor; hardcoded 1-critic ckpt |
| `scripts/profile_l3_speeds.py` | 24, 35, 38 | 2-arg ctor; hardcoded 1-critic ckpt |

Three scripts hardcode a pre-8/16 checkpoint (July 13/14, `hidden_dim=64`) that has
no `l3_critic_local`/`_global` keys and the wrong width. They must be repointed at a
current 2-critic checkpoint:
`output/8_16/training/five_station_15t_hierarchical/20260816_140621/checkpoints/best.pt`
(`l3_actor`/`l3_critic_local`/`l3_critic_global`, `hidden_dim=128`).

## E1 rerun

```bash
python scripts/bench_l3_latency.py --scenario five_station_15t \
  --l3 output/8_16/training/five_station_15t_hierarchical/20260816_140621/checkpoints/best.pt
```

Expect: SA 300-iteration re-solve vs per-section worker forward, both linear in
n_sections (4). Record the numbers in `coding_log`.

## Verification

- `python -c "import ast,sys; [ast.parse(open(f).read()) for f in FILES]"` — syntax.
- Rerun the broken scripts' import/entry paths far enough to confirm no
  `TypeError`/`AttributeError`/`KeyError` from the old API.
- E1 produces numbers, not a traceback.

## Out of scope

- Retraining the convergence protocol (depends on the provenance check).
- The SA-vs-FWA baseline scope question.
- `run_flat_mappo.py`'s own `self.critic` — that is a separate flat-BaselineTrainer
  class, not `L3WorkerTrainer`; unrelated, leave alone.
