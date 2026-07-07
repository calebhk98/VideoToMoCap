"""Unit tests for Pipeline 2 (motion_model): config, registry, data prep, trainers.

GPU-free by construction -- the `noop` trainer exercises the whole prepare->train
flow with no torch/weights/repo, exactly as the HMR `noop` backend does for
Pipeline 1. Runs under pytest OR directly: `python tests/test_motion_model.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import data
from motion_model.config import MotionModelConfig, load_config
from motion_model.trainers import TrainerError, available_methods, get_trainer


def assert_raises(exc, fn):
    """Tiny pytest.raises stand-in so tests run under plain `python` too."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _fake_dataset(root: Path, n: int = 4) -> Path:
    """Fabricate a minimal Pipeline 1 output: amass/*.npz + index.json with a split."""
    amass = root / "amass"
    amass.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(n):
        cid = f"clip{i:02d}"
        np.savez(amass / f"{cid}.npz", poses=np.zeros((40, 156), np.float32),
                 trans=np.zeros((40, 3), np.float32), betas=np.zeros(16, np.float32),
                 mocap_framerate=np.float32(20.0))
        clips.append({"clip_id": cid, "n_frames": 40, "fps": 20.0,
                      "split": "val" if i == 0 else "train",
                      "action_cluster": i % 2})
    (root / "index.json").write_text(json.dumps({"clips": clips}))
    return root


def _cfg(tmp: Path, **kw) -> MotionModelConfig:
    return MotionModelConfig(dataset_dir=_fake_dataset(tmp / "dataset"),
                             work_root=tmp / "work", method=kw.pop("method", "noop"), **kw)


def test_registry_lists_all_methods():
    methods = available_methods()
    for expected in ("momask", "mdm", "protomotions", "closd", "noop"):
        assert expected in methods, methods


def test_unknown_method_fails_loud():
    assert_raises(TrainerError, lambda: get_trainer(MotionModelConfig(method="nope")))


def test_config_validation_rejects_bad_enums():
    assert_raises(ValueError, lambda: MotionModelConfig(conditioning="sometimes"))
    assert_raises(ValueError, lambda: MotionModelConfig(simulator="unity"))
    assert_raises(ValueError, lambda: MotionModelConfig(algorithm="magic"))


def test_config_roundtrip_yaml(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("method: momask\nnum_steps: 5\nconditioning: action\n")
        cfg = load_config(p)
        assert cfg.method == "momask" and cfg.num_steps == 5 and cfg.conditioning == "action"


def test_load_config_rejects_unknown_keys():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("method: noop\nnot_a_key: 1\n")
        assert_raises(ValueError, lambda: load_config(p))


def test_fps_warning_only_off_multiples_of_20():
    assert data.fps_warning(20) is None
    assert data.fps_warning(40) is None
    assert data.fps_warning(30) is not None


def test_noop_prepare_builds_humanml3d_skeleton():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        out = get_trainer(cfg).prepare()
        assert (out / "train.txt").exists() and (out / "val.txt").exists()
        # split came from the dataset: 1 val, 3 train
        assert out.joinpath("train.txt").read_text().split() == ["clip01", "clip02", "clip03"]
        assert len(list((out / "texts").glob("*.txt"))) == 4
        assert len(list((out / "amass_copy").glob("*.npz"))) == 4


def test_action_conditioning_writes_cluster_labels():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), conditioning="action")
        out = get_trainer(cfg).prepare()
        text = (out / "texts" / "clip01.txt").read_text()
        assert "action 1" in text, text  # clip01 has action_cluster == 1


def test_noop_train_writes_checkpoint_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        trainer = get_trainer(cfg)
        trainer.prepare()
        save = trainer.train()
        assert (save / "model_final.pt").exists()
        manifest = json.loads((save / "train_manifest.json").read_text())
        assert manifest["n_clips"] == 4 and manifest["synthetic"] is True


def test_missing_dataset_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = MotionModelConfig(dataset_dir=Path(tmp) / "nope", work_root=Path(tmp) / "w", method="noop")
        assert_raises(FileNotFoundError, lambda: get_trainer(cfg).prepare())


def test_real_trainers_require_repo_before_running():
    # Generators/controllers must fail loud (not silently no-op) without their repo.
    with tempfile.TemporaryDirectory() as tmp:
        for method in ("momask", "mdm", "protomotions", "closd"):
            cfg = _cfg(Path(tmp), method=method)
            assert_raises(TrainerError, lambda: get_trainer(cfg).train())


def test_feature_formats_are_declared():
    fmts = {m: get_trainer(MotionModelConfig(method=m)).feature_format for m in available_methods()}
    assert fmts["momask"] == "humanml3d_263"
    assert fmts["protomotions"] == "amass_smplh"
    assert fmts["noop"] == "synthetic"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} motion_model tests passed")


if __name__ == "__main__":
    _run_all()
