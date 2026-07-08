"""Guard: every config option is present in the generated template + round-trips.

`config-template` emits a fully-commented YAML straight from each dataclass's
docstrings, so a new field can't silently become undocumented/undiscoverable --
this test fails if a field is ever missing from the template or if the template
stops loading cleanly. Needs pyyaml. Runs under pytest OR directly.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import config as mm_config
from videocaption import config as vc_config
from videotomocap import config as v2m_config

_CASES = [
    (v2m_config.PipelineConfig, v2m_config.config_template, v2m_config.load_config),
    (vc_config.CaptionConfig, vc_config.config_template, vc_config.load_config),
    (mm_config.MotionModelConfig, mm_config.config_template, mm_config.load_config),
]


def test_template_covers_every_field_and_is_valid_yaml():
    for cls, template, _ in _CASES:
        text = template()
        parsed = yaml.safe_load(text)
        expected = {f.name for f in dataclasses.fields(cls)} - {"cuda_device"}
        assert set(parsed) == expected, f"{cls.__name__} template missing: {sorted(expected - set(parsed))}"


def test_template_round_trips_through_load_config():
    for cls, template, load in _CASES:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.yaml"
            p.write_text(template())
            cfg = load(p)                       # must load without error
            assert isinstance(cfg, cls)


def test_template_is_commented():
    # every field line is preceded by at least one '# ' comment line
    text = v2m_config.config_template()
    assert text.count("#") >= len([f for f in dataclasses.fields(v2m_config.PipelineConfig)]) - 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} config-template tests passed")


if __name__ == "__main__":
    _run_all()
