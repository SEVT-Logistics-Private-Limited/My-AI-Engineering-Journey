import pytest
import yaml

from pickplace.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def _write(tmp_path, mutate):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    mutate(data)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_default_config_loads(cfg):
    assert cfg.camera.width > 0 and len(cfg.robot.home_joints) == 7


def test_unknown_key_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["camera"].update(fovv=10))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)


def test_missing_key_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["grasp"].pop("lift_height"))
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(path)


def test_invalid_value_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["rl"].update(obs_source="lidar"))
    with pytest.raises(ConfigError, match="obs_source"):
        load_config(path)


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent.yaml")


def test_reward_that_permits_hovering_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["rl"]["reward"].update(step_penalty=0.0))
    with pytest.raises(ConfigError, match="hover"):
        load_config(path)


def test_success_threshold_above_baseline_lift_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["task"].update(success_lift=0.25))   # baseline lifts 0.20
    with pytest.raises(ConfigError, match="success_lift"):
        load_config(path)


def test_default_config_has_place_section(cfg):
    from pickplace.rewards import place_failure_threshold

    assert cfg.place.goal_tolerance < cfg.place.min_distance
    w = cfg.place.rl.reward
    assert w.fail_penalty > place_failure_threshold(w, cfg.place.rl.gamma, cfg.task.success_lift)


def test_failure_threshold_depends_on_shaping(cfg):
    import dataclasses

    from pickplace.rewards import place_failure_threshold, place_potential_max

    w, gamma, lift = cfg.place.rl.reward, cfg.place.rl.gamma, cfg.task.success_lift
    dense = place_failure_threshold(w, gamma, lift)
    assert dense == pytest.approx((w.step_penalty + w.goal_weight * w.distance_ref
                                   + w.lower_weight * w.lower_ref) / (1 - gamma))
    wp = dataclasses.replace(w, shaping="potential")
    assert place_failure_threshold(wp, gamma, lift) == pytest.approx(
        w.step_penalty / (1 - gamma) + place_potential_max(wp, lift))


def test_unknown_shaping_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["place"]["rl"]["reward"].update(shaping="sparse"))
    with pytest.raises(ConfigError, match="shaping"):
        load_config(path)


def test_goal_tolerance_must_be_below_min_distance(tmp_path):
    path = _write(tmp_path, lambda d: d["place"].update(goal_tolerance=0.12))
    with pytest.raises(ConfigError, match="goal_tolerance"):
        load_config(path)


def test_goal_must_fit_inside_spawn_region(tmp_path):
    path = _write(tmp_path, lambda d: d["place"].update(min_distance=0.3, goal_tolerance=0.05))
    with pytest.raises(ConfigError, match="fit inside the spawn region"):
        load_config(path)


def test_place_reward_that_permits_hovering_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["place"]["rl"]["reward"].update(step_penalty=0.0))
    with pytest.raises(ConfigError, match="hover"):
        load_config(path)


def test_place_reward_where_failing_early_pays_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["place"]["rl"]["reward"].update(fail_penalty=5.0))
    with pytest.raises(ConfigError, match="failing beats trying"):
        load_config(path)


def test_missing_place_section_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d.pop("place"))
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(path)


def test_curriculum_probabilities_must_sum_to_at_most_one(tmp_path):
    path = _write(tmp_path, lambda d: d["place"]["rl"]["curriculum"].update(start_grasped_prob=0.6,
                                                                             start_above_goal_prob=0.6))
    with pytest.raises(ConfigError, match="curriculum"):
        load_config(path)
