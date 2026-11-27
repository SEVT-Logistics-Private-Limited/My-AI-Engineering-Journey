import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pickplace.env_place import OBS_DIM, PickPlaceEnv, place_potential, register


class ScriptedPlacer:
    """Hand-written policy through the action API: proves the task is solvable,
    and with ``release_height``/``hold`` it acts out the reward hacks to reject."""

    def __init__(self, release_height: float = 0.02, hold: bool = False):
        self.release_height, self.hold = release_height, hold
        self.closing = self.released = False

    def __call__(self, obs):
        ee, width, obj, goal, lifted = obs[:3], obs[3], obs[4:7], obs[10:13], obs[16]
        rel = obj - ee
        if not lifted:
            if not self.closing and width > 0.06:
                if np.linalg.norm(rel[:2]) > 0.01:
                    return np.r_[np.clip(rel[:2] / 0.03, -1, 1), np.clip((obj[2] + 0.06 - ee[2]) / 0.03, -1, 1), -1]
                if abs(rel[2]) > 0.008:
                    return np.r_[np.clip(rel / 0.03, -1, 1), -1]
            self.closing = True
            return np.r_[0, 0, 1 if width < 0.045 else 0, 1]
        if self.released:
            return np.r_[0, 0, 1, -1]
        d = goal[:2] - obj[:2]
        if np.linalg.norm(d) > 0.004:
            return np.r_[np.clip(d / 0.03, -1, 1), np.clip((0.12 - obj[2]) / 0.03, -1, 1), 1]
        target_z = goal[2] + self.release_height
        if abs(obj[2] - target_z) > 0.004:
            return np.r_[np.clip(d / 0.03, -1, 1), np.clip((target_z - obj[2]) / 0.03, -1, 1), 1]
        if self.hold:
            return np.r_[0, 0, 0, 1]
        self.released = True
        return np.r_[0, 0, 0, -1]


def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rewards, done = [], False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        rewards.append(r)
        done = terminated or truncated
    return np.array(rewards), terminated, info


@pytest.fixture(scope="module")
def env(cfg):
    e = PickPlaceEnv(cfg)
    yield e
    e.close()


def test_passes_gymnasium_checker(cfg):
    e = PickPlaceEnv(cfg)
    try:
        check_env(e, skip_render_check=True)
    finally:
        e.close()


def test_observation_shape_and_seeded_reset(env):
    a, info = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    assert a.shape == (OBS_DIM,) == (17,)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a[16] == 0.0 and not info["is_success"]


def test_goal_matches_the_baseline_for_the_same_seed(cfg, env):
    from pickplace.task import sample_place_goal

    obs, _ = env.reset(seed=30000)
    rng = np.random.default_rng(30000)
    pick = np.array([rng.uniform(*cfg.workspace.object_x), rng.uniform(*cfg.workspace.object_y)])
    rng.uniform(*cfg.object.yaw_range)            # the baseline's yaw draw
    np.testing.assert_allclose(obs[10:12], sample_place_goal(rng, pick, cfg), atol=1e-6)
    np.testing.assert_allclose(obs[4:6], pick, atol=2e-3)


@pytest.mark.parametrize("obs_source,curriculum,seeds", [
    ("ground_truth", False, (0, 1, 2)), ("ground_truth", True, (3, 4)), ("perception", False, (5,)),
])
def test_task_is_solvable(cfg, obs_source, curriculum, seeds):
    e = PickPlaceEnv(cfg, obs_source=obs_source, curriculum=curriculum)
    try:
        for seed in seeds:
            rewards, terminated, info = run_episode(e, ScriptedPlacer(), seed)
            assert info["is_success"] and terminated, (obs_source, seed, info["outcome"])
            assert info["distance_moved"] >= cfg.place.min_distance
            assert rewards[-1] > cfg.place.rl.reward.success_bonus / 2
    finally:
        e.close()


def test_curriculum_rejects_perception(cfg):
    with pytest.raises(ValueError, match="curriculum"):
        PickPlaceEnv(cfg, obs_source="perception", curriculum=True)


def test_registered_with_gymnasium():
    register()
    e = gym.make("PickPlace-v0")
    obs, _ = e.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    e.close()


# ------------------------------------------------------------------ reward hacking
def test_hovering_with_the_cube_at_the_goal_never_pays(cfg, env):
    rewards, terminated, info = run_episode(env, ScriptedPlacer(hold=True), seed=1)
    assert not info["is_success"] and not terminated
    assert info["outcome"] == "not_released" and info["distance_to_goal"] < cfg.place.goal_tolerance
    assert np.all(rewards[-50:] < 0)           # every step of hovering costs reward


def test_cube_on_the_goal_without_lifting_is_not_success(cfg, env):
    """Pushing or sliding: the cube reaches the goal without ever being lifted."""
    obs, _ = env.reset(seed=2)
    env.scene.place_object(obs[10:12], 0.0)
    for _ in range(20):
        obs, r, terminated, _, info = env.step(np.array([0, 0, 1.0, -1.0]))
        assert not terminated and r < 0
    assert info["outcome"] == "not_lifted" and info["distance_to_goal"] < 0.005


def test_throwing_is_a_terminal_failure(cfg, env):
    rewards, terminated, info = run_episode(env, ScriptedPlacer(release_height=0.10), seed=0)
    assert terminated and not info["is_success"]
    assert info["outcome"] == "released_too_high"
    assert rewards[-1] < -cfg.place.rl.reward.fail_penalty / 2


def _potential(cfg, **state):
    return place_potential(cfg.place.rl.reward, cfg.task.success_lift, **state)


def _loop_return(cfg, states):
    """Total shaping reward around a closed loop of states (first == last)."""
    gamma = cfg.place.rl.gamma
    phis = [_potential(cfg, **s) for s in states]
    return sum(gamma * b - a for a, b in zip(phis, phis[1:]))


def test_lifting_again_and_again_earns_nothing(cfg):
    held = dict(lifted=True, grasped=True, reach_dist=0.0, goal_dist=0.2)
    loop = [dict(held, lift=0.06, bottom_height=0.06), dict(held, lift=0.0, bottom_height=0.0),
            dict(held, lift=0.06, bottom_height=0.06)]
    assert _loop_return(cfg, loop * 5) < 0
    # Once lifted, the lift height away from the goal no longer changes the potential.
    assert _potential(cfg, **loop[0]) == _potential(cfg, **loop[1])


def test_putting_down_and_picking_up_again_earns_nothing(cfg):
    on_goal = dict(lifted=True, reach_dist=0.0, lift=0.0, goal_dist=0.01, bottom_height=0.0)
    loop = [dict(on_goal, grasped=True), dict(on_goal, grasped=False),
            dict(on_goal, grasped=True, lift=0.1, bottom_height=0.1), dict(on_goal, grasped=True)]
    assert _loop_return(cfg, loop) < 0


def test_no_loop_of_states_pays(cfg):
    """phi >= 0 and potential-based: any closed loop returns (gamma - 1) * sum(phi) <= 0."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        states = [dict(lifted=bool(rng.random() < 0.5), grasped=bool(rng.random() < 0.5),
                       reach_dist=rng.uniform(0, 0.6), lift=rng.uniform(0, 0.2),
                       goal_dist=rng.uniform(0, 0.6), bottom_height=rng.uniform(0, 0.25))
                  for _ in range(rng.integers(2, 8))]
        assert all(_potential(cfg, **s) >= 0 for s in states)
        assert _loop_return(cfg, states + states[:1]) <= 1e-12


def test_progress_through_the_stages_increases_the_potential(cfg):
    stages = [
        dict(lifted=False, grasped=False, reach_dist=0.3, lift=0.0, goal_dist=0.2, bottom_height=0.0),
        dict(lifted=False, grasped=False, reach_dist=0.0, lift=0.0, goal_dist=0.2, bottom_height=0.0),
        dict(lifted=False, grasped=True, reach_dist=0.0, lift=0.0, goal_dist=0.2, bottom_height=0.0),
        dict(lifted=True, grasped=True, reach_dist=0.0, lift=0.05, goal_dist=0.2, bottom_height=0.05),
        dict(lifted=True, grasped=True, reach_dist=0.0, lift=0.10, goal_dist=0.0, bottom_height=0.10),
        dict(lifted=True, grasped=True, reach_dist=0.0, lift=0.00, goal_dist=0.0, bottom_height=0.0),
    ]
    phis = [_potential(cfg, **s) for s in stages]
    assert all(b > a for a, b in zip(phis, phis[1:])), phis
