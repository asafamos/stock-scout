import pytest
from core.pipeline_runner import _normalize_config
from core.config import get_config, Config


def test_normalize_config_with_dataclass():
    cfg = get_config()
    out = _normalize_config(cfg)
    assert isinstance(out, dict)
    # Expect keys from Config.to_dict (lowercase)
    assert 'universe_limit' in out
    assert 'smart_scan' in out


def test_normalize_config_with_dict():
    inp = {"A": 1}
    out = _normalize_config(inp)
    assert isinstance(out, dict)
    assert out["A"] == 1


def test_normalize_config_unsupported_object_raises():
    class Foo: pass
    with pytest.raises(TypeError):
        _normalize_config(Foo())
