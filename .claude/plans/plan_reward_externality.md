# Plan: internalize the moving-block externality in the L3 reward

Status: draft — noted for next step (alongside 20-train test).

## Problem

Current L3 reward is LOCAL (own arrival deviation) + a weak shared team scalar.
Learned optimises ≈ SA's objective (own punctuality) → cannot beat the per-transit
oracle. The team term is not credited per-action, so no coordination gradient.

## Root: moving-block externality not internalised

When a train runs slow / arrives late, the FOLLOWING train is headway-held — the
follower's wait penalises the FOLLOWER, not the leader that caused it. So the leader
has no incentive to avoid creating follower delay (or to pace to reduce system impact).

## Fix (option 1 — most direct, physically grounded)

Charge the LEADER for the headway wait it imposes on the follower:
- In `section_progress`, when a follower's `wait` > 0 due to the front train, record
  that wait as attributable to the front train.
- Subtract a share of the follower's wait (× a coefficient) from the LEADER's reward
  at its terminal (corridor end).

This internalises the externality → the leader learns to pace so it doesn't jam the
follower, i.e. real coordination (the classic railway "front-train perturbation →
following delay" mechanism).

## Alternatives
- Option 2: per-action team credit (credit a train when its action reduces future
  system delay — needs lookahead, harder).
- Option 3: pure team objective (reward = system ontime rate + system delay) — hardest
  credit assignment.

## Note
20-train environment is NECESSARY but NOT sufficient: it creates conflicts (scarcity);
the reward fix creates the coordination INCENTIVE. Both needed for learned to beat SA
on system metrics.
