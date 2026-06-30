#!/usr/bin/env python3
"""Flat MAPPO training script (ablation: no L1 action mask).

Same as run_train.py but disables the Layer-1 action mask, so the policy
must learn boundary constraints from reward alone (held=-3.0 for infeasible
actions reaching Layer-1 bounds violation).

Usage:
    python -m src.experiment.run_train_flat --episodes 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim
from src.experiment.run_train import train, HP
from src.experiment.scenario_loader import load_scenario
from src.control.action_mask import ENTRY_CHOICES


class FlatFiveStationEnv(FiveStationEnv):
    """FiveStationEnv with Layer-1 action mask disabled.

    The policy sees ALL actions as legal. Infeasible actions (violating
    timetable bounds) are caught by _build_requests() and receive held=-3.0.
    """

    def _action_masks(self, pending_ids: List[str]):
        masks = {}
        for tid in pending_ids:
            train = self.cfg.trains[tid]
            st = self._journeys[tid].current_station
            platforms = train.candidate_platforms.get(st, [])
            masks[tid] = (
                [True] * len(platforms),
                [True] * len(ENTRY_CHOICES),
            )
        return masks


def main():
    parser = argparse.ArgumentParser(description="Flat MAPPO (no L1 mask)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lr", type=float, default=HP["lr"])
    args = parser.parse_args()
    args.env = "flat_mappo"  # separate output from main experiment

    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)

    def env_factory(seed: int) -> FlatFiveStationEnv:
        return FlatFiveStationEnv(
            cfg=cfg, seed=seed, batch_window_sec=120.0,
        )

    train(env_factory, args, obs_dim=obs_dim)


if __name__ == "__main__":
    main()
