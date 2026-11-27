"""Reward math for pick-and-place, shared by the environment and config validation.

Pure functions with no simulator and no config import, so ``config.py`` can use
them to validate reward settings and every term can be unit-tested.

The staged potential phi(s) >= 0 rises through the task:

    before lifted   reach_weight * (distance_ref - hand-to-cube distance)
                    + while grasped: grasp_bonus + lift_weight * lift (capped at success_lift)
    after lifted    the maximum of everything above
                    + goal_weight * (distance_ref - cube-to-goal distance)
                    + lower_weight * (lower_ref - cube height), faded in within lower_radius of the goal

Once lifted, phi no longer depends on the grasp or the lift height, so neither
re-lifting nor putting the cube down and picking it up again raises it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import PlaceRewardConfig


def place_potential(w: "PlaceRewardConfig", success_lift: float, *, lifted: bool, grasped: bool,
                    reach_dist: float, lift: float, goal_dist: float, bottom_height: float) -> float:
    """Shaping potential phi(s) >= 0, increasing through the task's stages."""
    pre_lift_max = w.reach_weight * w.distance_ref + w.grasp_bonus + w.lift_weight * success_lift
    if not lifted:
        phi = w.reach_weight * max(0.0, w.distance_ref - reach_dist)
        if grasped:
            phi += w.grasp_bonus + w.lift_weight * float(np.clip(lift, 0.0, success_lift))
        return phi
    near_goal = max(0.0, 1.0 - goal_dist / w.lower_radius)
    return (pre_lift_max
            + w.goal_weight * max(0.0, w.distance_ref - goal_dist)
            + w.lower_weight * near_goal * max(0.0, w.lower_ref - max(0.0, bottom_height)))


def place_potential_max(w: "PlaceRewardConfig", success_lift: float) -> float:
    """phi at the goal, on the floor: the largest value ``place_potential`` can return."""
    return (w.reach_weight * w.distance_ref + w.grasp_bonus + w.lift_weight * success_lift
            + w.goal_weight * w.distance_ref + w.lower_weight * w.lower_ref)


def place_step_reward(w: "PlaceRewardConfig", gamma: float, success_lift: float, phi: float, phi_prev: float) -> float:
    """Per-step reward before the terminal success bonus or fail penalty.

    dense      -step_penalty - (phi_max - phi): every step costs at least step_penalty,
               and being further along the task costs less on every following step, so
               value differences scale with 1 / (1 - gamma)
    potential  -step_penalty + gamma * phi - phi_prev: policy-invariant shaping, but a value
               difference between two states is only their difference in phi
    """
    if w.shaping == "dense":
        return -w.step_penalty - (place_potential_max(w, success_lift) - phi)
    return -w.step_penalty + gamma * phi - phi_prev


def place_failure_threshold(w: "PlaceRewardConfig", gamma: float, success_lift: float) -> float:
    """Smallest fail penalty that never makes a terminal failure attractive.

    Terminal failures need a lifted cube. Dense: after the lift a step costs at most
    step_penalty + goal_weight * distance_ref + lower_weight * lower_ref, so never succeeding
    costs that / (1 - gamma). Potential: never succeeding costs step_penalty / (1 - gamma)
    plus the potential that decays away, while terminating keeps it, so add phi's maximum.
    """
    if w.shaping == "dense":
        return (w.step_penalty + w.goal_weight * w.distance_ref + w.lower_weight * w.lower_ref) / (1 - gamma)
    return w.step_penalty / (1 - gamma) + place_potential_max(w, success_lift)


__all__ = ["place_failure_threshold", "place_potential", "place_potential_max", "place_step_reward"]
