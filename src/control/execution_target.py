"""Layer 2 to Layer 3 execution target types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionTarget:
    train_id: str
    station_id: str
    allocated_platform: str
    route_id: str
    platform_edge: str
    t_throat_entry: float
    t_platform_arrival: float
    t_platform_departure: float
    t_throat_exit: float
    pass_flag: bool
    dwell_min: float
    destination_platform_edge: Optional[str] = None
    destination_dwell_min: float = 0.0
