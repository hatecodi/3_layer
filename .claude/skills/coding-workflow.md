---
name: coding-workflow
description: |
  Enforces a plan-then-code-then-log discipline. Before any code change, write a
  plan and verify alignment with project design docs. After each modification,
  append a timestamped entry to coding_log. Designed for multi-agent
  collaboration — other Claude instances (or human reviewers) can review the log
  and reconstructed plans to understand what changed and why.
type: project
---

# Coding workflow discipline

This skill defines the mandatory workflow for **every code modification** in this
project. Follow these rules without exception.

## Rule 1: Plan before coding

Before calling any Edit or Write tool on a `.py`, `.yaml`, `.xml`, or
`.sumocfg` file:

1. **Write the plan.** Use Write to create/update a plan file that states:
   - What will change (files and functions/classes)
   - Why the change is needed
   - How it aligns with the project design in `three_layer_railway_marl_fixed_1.md`
     and `railway_marl_spec.md`
   - Expected outcome and pass criteria

2. **Check alignment with project goals.** Verify the plan against these hard
   constraints:
   - No station name, train ID, or platform ID hardcoded in Python
   - Observation always 74-dim (or derived from config, not magic numbers)
   - Action masking enforces Layer 1 bounds, not reward penalties
   - Local critic input is `(K_max+1) × 74 = 444`, not global
   - Fallback re-verifies all bounds via `RebuildFn`
   - Adding trains/stations requires only config/SUMO file changes, never Python changes

3. **If the plan does NOT pass the alignment check**, revise it before touching
   any code. Do not proceed with an invalid plan.

4. **For trivial single-line fixes** (typos, obvious bugs, formatting), the plan
   can be one sentence — but it must still be written.

## Rule 2: Log after coding

After completing a code modification (one or more related Edit/Write calls that
form a coherent change):

1. **Append to `coding_log`** with the following format:

   ```
   [YYYY-MM-DD HH:MM]
   Task: <one-line summary of what was done>
   Files:
     <path> — <brief description of change>
   Rationale: <why this change, how it connects to the plan>
   Status: <DONE | PARTIAL — what remains>
   ```

2. **Keep entries atomic.** One log entry per completed task, not per file edit.

3. **Write logs so another AI can understand the context.** Include enough detail
   that a fresh Claude instance can read `coding_log` and understand the project
   trajectory without reading every commit.

## Rule 3: Before starting a session

When starting work in this repository:

1. Read `CLAUDE.md` to understand the project architecture.
2. Read `coding_log` to see recent changes and current status.
3. Read the last plan file (if any) to understand the current task.
4. Read the design documents (`three_layer_railway_marl_fixed_1.md` and
   `railway_marl_spec.md`) if touching core architecture.

## Plan file convention

Store plans in `.claude/plans/` as `plan_<short-task-name>.md`. Each plan file
must contain:

```markdown
# Plan: <task name>
Date: YYYY-MM-DD
Status: <draft | in_progress | done | abandoned>

## What
<What will change — files, functions, classes>

## Why
<Why this change is needed>

## Alignment check
- [ ] No hardcoded IDs: <OK / issue>
- [ ] Observation 74-dim: <OK / issue / N/A>
- [ ] Action mask enforces bounds: <OK / issue / N/A>
- [ ] Local critic 444-dim: <OK / issue / N/A>
- [ ] Fallback re-verifies bounds: <OK / issue / N/A>
- [ ] Config-only expansion: <OK / issue / N/A>

## Pass criteria
<How will we know this worked?>
```
