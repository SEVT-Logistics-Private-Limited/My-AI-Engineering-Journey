import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pickplace.env import OBS_DIM, PickLiftEnv, register


def scripted_expert(obs):
    """Hand-written policy used to prove the task is solvable through the action API."""
    ee, width, obj = obs[:3], obs[3], obs[4:7]
    rel = obj - ee
    if np.linalg.norm(rel[:2]) > 0.01 and width > 0.06:
        return np.r_[np.clip(rel[:2] / 0.03, -1, 1), np.clip((obj[2] + 0.06 - ee[2]) / 0.03, -1, 1), -1]
    if abs(rel[2]) > 0.008 and width > 0.06:
        return np.r_[np.clip(rel / 0.03, -1, 1), -1]
    return np.r_[0, 0, 1 if width < 0.045 else 0, 1]


@pytest.fixture
def env(cfg):
    e = PickLiftEnv(cfg)
    yield e
    e.close()


def test_passes_gymnasium_checker(env):
    check_env(env, skip_render_check=True)


def test_reset_is_deterministic_under_seed(env):
    a, _ = env.reset(seed=3)
    b, _ = env.reset(seed=3)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a.shape == (OBS_DIM,)


def test_rejects_bad_action_shape(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(3))


def test_end_effector_stays_in_workspace(env, cfg):
    env.reset(seed=0)
    for _ in range(40):
        obs, *_ = env.step(np.array([1.0, 1.0, -1.0, -1.0]))
    assert np.all(obs[:3] >= np.array(cfg.workspace.ee_low) - 0.02)
    assert np.all(obs[:3] <= np.array(cfg.workspace.ee_high) + 0.02)


@pytest.mark.parametrize("obs_source,episodes", [("ground_truth", 5), ("perception", 2)])
def test_task_is_solvable(cfg, obs_source, episodes):
    env = PickLiftEnv(cfg, obs_source=obs_source)
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            done, info, total = False, {}, 0.0
            while not done:
                obs, r, terminated, truncated, info = env.step(scripted_expert(obs))
                total += r
                done = terminated or truncated
            assert info["is_success"], f"{obs_source} episode {ep} failed"
            assert terminated
    finally:
        env.close()


def test_registered_with_gymnasium():
    register()
    e = gym.make("PickLift-v0")
    obs, _ = e.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    e.close()


def _rollout(env, seed):
    obs, _ = env.reset(seed=seed)
    traj = [obs]
    for _ in range(30):
        obs, *_ = env.step(scripted_expert(obs))
        traj.append(obs)
    return np.array(traj)


def test_episode_depends_only_on_its_seed(cfg):
    """Same seed -> same trajectory, whatever ran before (PyBullet caches contacts across resets)."""
    fresh = PickLiftEnv(cfg)
    used = PickLiftEnv(cfg)
    try:
        expected = _rollout(fresh, seed=5)
        for s in range(1, 5):
            _rollout(used, seed=s)
        np.testing.assert_allclose(_rollout(used, seed=5), expected, atol=1e-6)
    finally:
        fresh.close()
        used.close()
