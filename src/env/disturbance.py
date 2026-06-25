"""Disturbance module — simplified: no delay injection, only deadlock threshold.

In the simplified design, Layer 2 does NOT inject or recover from disturbances.
Trains compete only on platform and throat resources at each station.
Conflicts arise from natural timetable overlaps, not from injected delays.
"""

DEADLOCK_THRESHOLD = 15  # global NO_LEGAL_ACTION consecutive steps
