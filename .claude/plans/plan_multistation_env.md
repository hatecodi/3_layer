# Plan: Multi-station railway scheduling environment

Date: 2026-06-03
Status: in_progress

## Network layout

```
A ←1200m/90s→ B ←2400m/180s→ C ←1200m/90s→ D ←1200m/90s→ E
```

| Station | Platforms | Dwell (stop) | Notes |
|---------|-----------|-------------|-------|
| A | 3 | 60s | existing |
| B | 3 | 30s | existing |
| C | 4 | 60s | large station, same dwell as A |
| D | 2 | 15s | **intentional bottleneck** |
| E | 3 | 30s | same as B |

Corridors: A_B=90s, B_C=180s (2x), C_D=90s, D_E=90s

## Trains

Keep existing 8+1 trains. Direction:
- Up trains (A→E): K641, D133, G5050, K667, K999
  - Route: A → B → C → D → E
  - K999 delayed from upstream
- Down trains (E→A): G1408, G5060, Z386, K1666
  - Route: E → D → C → B → A

Timetable generated from distance / avg_speed ≈ 1200m/13.3m/s ≈ 90s per segment.

## Multi-station scheduling loop

Each train needs scheduling at EVERY station on its route:
1. Reset: all trains pending at their origin station
2. Each step: pending trains at each station get scheduled (origin + en-route)
3. After grant: train moves to next station after corridor transit time
4. Corridor disturbances may delay transit
5. D station (2 platforms) creates natural conflicts → L2 must optimize

## Key changes from single-station env

| Change | Before | After |
|--------|--------|-------|
| Stations per train | 1 (origin only) | up to 5 |
| Steps per episode | 1-2 | ~5×9=45+ |
| State transitions | None (one-shot) | Corridor delay → next station arrival |
| Bottleneck | None | D station (2 platforms) |
| L2 benefit | Limited (bandit) | Real scheduling optimization |

## New folder: `multistation/`

```
multistation/
  configs/
    railway_resources.yaml
    timetable.yaml
  sumo/
    net/
    routes/
  src/
    (reuse or import from parent project)
  scripts/
    verify_multistation.py
    run_heuristic.py
    run_train.py
```

## Phase breakdown

1. Generate timetable (distance-based arrival times)
2. Create railway_resources.yaml for 5-station network
3. Extend sumo_env to multi-station scheduling
4. Run heuristic baseline on multi-station
5. Run MAPPO training on multi-station
