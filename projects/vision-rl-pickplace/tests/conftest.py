import pytest

from pickplace.config import load_config
from pickplace.sim import Scene


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def scene(cfg):
    s = Scene(cfg)
    yield s
    s.close()
