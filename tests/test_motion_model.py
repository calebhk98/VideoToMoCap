"""Unit tests for Pipeline 2 (motion_model): config, registry, data prep, trainers.

GPU-free by construction -- the `noop` trainer exercises the whole prepare->train
flow with no torch/weights/repo, and every real trainer's command construction is
asserted by capturing `_run_cmd` (so the upstream flags are actually pinned, not
just "fails without a repo"). Runs under pytest OR directly:
`python tests/test_motion_model.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_model import cli, data, features
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
                      "split": "val" if i == 0 else "train", "action_cluster": i % 2})
    (root / "index.json").write_text(json.dumps({"clips": clips}))
    return root


def _cfg(tmp: Path, **kw) -> MotionModelConfig:
    return MotionModelConfig(dataset_dir=_fake_dataset(tmp / "dataset"),
                             work_root=tmp / "work", method=kw.pop("method", "noop"), **kw)


def _fake_repo(tmp: Path) -> Path:
    """A directory that passes _require_repo (contents don't matter for the tests)."""
    repo = tmp / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def _capture(trainer) -> list:
    """Override the instance _run_cmd to record (cmd, cwd) instead of shelling out."""
    calls = []
    trainer._run_cmd = lambda cmd, cwd=None: calls.append(([str(c) for c in cmd], cwd))
    return calls


def _val(cmd: list, flag: str):
    """The token following ``flag`` in a command list, or None if absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# -- config + registry ------------------------------------------------------

def test_registry_lists_all_methods():
    for expected in ("momask", "mdm", "protomotions", "closd", "noop"):
        assert expected in available_methods()


def test_unknown_method_fails_loud():
    assert_raises(TrainerError, lambda: get_trainer(MotionModelConfig(method="nope")))


def test_config_validation_rejects_bad_enums():
    assert_raises(ValueError, lambda: MotionModelConfig(conditioning="sometimes"))
    assert_raises(ValueError, lambda: MotionModelConfig(simulator="unity"))
    assert_raises(ValueError, lambda: MotionModelConfig(algorithm="magic"))


def test_config_roundtrip_yaml():
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


def test_config_paths_and_to_dict():
    cfg = MotionModelConfig(method="momask", dataset_dir=Path("work/ds"))
    assert cfg.amass_dir == Path("work/ds/amass")
    assert cfg.index_path == Path("work/ds/index.json")
    assert cfg.prepared_dir.parts[-2:] == ("momask", "prepared")
    d = cfg.to_dict()
    assert d["method"] == "momask" and isinstance(d["dataset_dir"], str)


# -- data helpers -----------------------------------------------------------

def test_fps_warning_only_off_multiples_of_20():
    assert data.fps_warning(20) is None and data.fps_warning(40) is None
    assert data.fps_warning(30) is not None


def test_humanml3d_handoff_missing_vs_ready():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        missing = data.humanml3d_handoff(SimpleNamespace(tmr_repo=None, smpl_model=None), tmp)
        assert "mano.is.tue.mpg.de" in missing and "Mathux/TMR" in missing
        tmr = tmp / "tmr"; tmr.mkdir()
        model = tmp / "m.npz"; model.write_bytes(b"x")
        ready = data.humanml3d_handoff(SimpleNamespace(tmr_repo=tmr, smpl_model=model), tmp)
        assert "extracted offline" in ready


def _feature_cfg(tmp: Path):
    """A config with the offline feature assets (fake but present) wired up."""
    tmr = tmp / "tmr"; tmr.mkdir(parents=True, exist_ok=True)
    model = tmp / "smplh_neutral.npz"; model.write_bytes(b"x")
    return _cfg(tmp, method="momask", tmr_repo=tmr, smpl_model=model)


def test_features_has_assets():
    with tempfile.TemporaryDirectory() as tmp:
        assert features.has_assets(_feature_cfg(Path(tmp)))
        assert not features.has_assets(_cfg(Path(tmp) / "x", method="momask"))


def test_features_driver_matches_upstream_convention():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _feature_cfg(tmp)
        script = features._driver_script(cfg, tmp / "vecs")
        assert "joints_to_guofeats" in script          # drives TMR, not a reimplementation
        assert "joints[..., 0] *= -1" in script         # TMR's proper-rotation fix
        assert "num_betas=10" in script                 # neutral FK, DMPL skipped
        assert str(cfg.tmr_repo) in script and str(cfg.smpl_model) in script


def test_prepare_runs_feature_extraction_when_assets_present():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _feature_cfg(Path(tmp))
        captured = {}
        orig = features._run
        features._run = lambda c, script: captured.setdefault("script", script)
        try:
            get_trainer(cfg).prepare()   # momask prepare -> data.prepare_humanml3d -> features.extract
        finally:
            features._run = orig
        assert "joints_to_guofeats" in captured.get("script", "")


# -- noop end-to-end --------------------------------------------------------

def test_noop_prepare_builds_humanml3d_skeleton():
    with tempfile.TemporaryDirectory() as tmp:
        out = get_trainer(_cfg(Path(tmp))).prepare()
        assert out.joinpath("train.txt").read_text().split() == ["clip01", "clip02", "clip03"]
        assert len(list((out / "texts").glob("*.txt"))) == 4
        assert len(list((out / "amass_copy").glob("*.npz"))) == 4


def test_action_conditioning_writes_cluster_labels():
    with tempfile.TemporaryDirectory() as tmp:
        out = get_trainer(_cfg(Path(tmp), conditioning="action")).prepare()
        assert "action 1" in (out / "texts" / "clip01.txt").read_text()


def test_noop_train_writes_checkpoint_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = get_trainer(_cfg(Path(tmp)))
        trainer.prepare()
        save = trainer.train()
        assert (save / "model_final.pt").exists()
        assert json.loads((save / "train_manifest.json").read_text())["n_clips"] == 4


def test_missing_dataset_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = MotionModelConfig(dataset_dir=Path(tmp) / "nope", work_root=Path(tmp) / "w", method="noop")
        assert_raises(FileNotFoundError, lambda: get_trainer(cfg).prepare())


# -- command construction for the real trainers -----------------------------

def test_momask_train_command_and_staging():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, method="momask", repo=_fake_repo(tmp), num_steps=500)
        trainer = get_trainer(cfg)
        trainer.prepare()
        calls = _capture(trainer)
        trainer.train()
        cmd, cwd = calls[0]
        assert "train_vq.py" in cmd and cwd == cfg.repo
        assert "--data_root" not in cmd                    # dropped: hardcoded path upstream
        assert _val(cmd, "--name") == "mymotion_rvq"
        assert _val(cmd, "--dataset_name") == "t2m"
        assert _val(cmd, "--max_epoch") == "1"             # max(1, 500//1000) floors at 1
        assert (cfg.repo / "dataset" / "HumanML3D").is_symlink()   # data staged where the loader looks


def test_mdm_full_uses_resume_checkpoint_and_drops_bad_flags():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "prior.pt"; ckpt.write_bytes(b"x")
        cfg = _cfg(tmp, method="mdm", repo=_fake_repo(tmp), resume_checkpoint=ckpt)
        trainer = get_trainer(cfg)
        trainer.prepare()
        calls = _capture(trainer)
        trainer.train()
        cmd = calls[0][0]
        assert cmd[cmd.index("-m") + 1] == "train.train_mdm"
        assert _val(cmd, "--resume_checkpoint") == str(ckpt)
        for bad in ("--guidance_param", "--data_dir", "--unconstrained", "--lora_finetune"):
            assert bad not in cmd, bad


def test_mdm_lora_uses_starting_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = tmp / "prior.pt"; ckpt.write_bytes(b"x")
        cfg = _cfg(tmp, method="mdm", repo=_fake_repo(tmp), resume_checkpoint=ckpt, personalization="lora")
        trainer = get_trainer(cfg)
        trainer.prepare()
        calls = _capture(trainer)
        trainer.train()
        cmd = calls[0][0]
        assert "--lora_finetune" in cmd
        assert _val(cmd, "--starting_checkpoint") == str(ckpt)
        assert "--resume_checkpoint" not in cmd


def test_mdm_requires_resume_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), method="mdm", repo=_fake_repo(Path(tmp)))  # no resume_checkpoint
        assert_raises(TrainerError, lambda: get_trainer(cfg).train())


def test_protomotions_prepare_two_stage_and_no_output_flag():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        smpl = tmp / "smpl"; smpl.mkdir()
        cfg = _cfg(tmp, method="protomotions", repo=_fake_repo(tmp), smpl_model=smpl)
        trainer = get_trainer(cfg)
        calls = _capture(trainer)
        trainer.prepare()
        convert, package = calls[0][0], calls[1][0]
        assert any("convert_amass_to_proto.py" in t for t in convert) and "--output" not in convert
        assert _val(convert, "--humanoid-type") == "smpl" and _val(convert, "--output-fps") == "30"
        assert any("motion_lib.py" in t for t in package) and _val(package, "--device") == "cpu"
        assert _val(package, "--output-file").endswith("motions.pt")


def test_protomotions_train_command_and_experiment_path():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        smpl = tmp / "smpl"; smpl.mkdir()
        cfg = _cfg(tmp, method="protomotions", repo=_fake_repo(tmp), smpl_model=smpl)
        cfg.prepared_dir.mkdir(parents=True, exist_ok=True)
        (cfg.prepared_dir / "motions.pt").write_bytes(b"x")   # pretend prepare ran
        trainer = get_trainer(cfg)
        calls = _capture(trainer)
        trainer.train()
        cmd = calls[0][0]
        assert "--batch-size" in cmd                         # was missing -> would crash upstream
        assert _val(cmd, "--experiment-path") == "examples/experiments/masked_mimic/transformer.py"
        assert _val(cmd, "--simulator") == "isaaclab"


def test_protomotions_train_requires_motion_file():
    with tempfile.TemporaryDirectory() as tmp:
        smpl = Path(tmp) / "smpl"; smpl.mkdir()
        cfg = _cfg(Path(tmp), method="protomotions", repo=_fake_repo(Path(tmp)), smpl_model=smpl)
        assert_raises(TrainerError, lambda: get_trainer(cfg).train())


def test_closd_train_command_and_resume_toggle():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, method="closd", repo=_fake_repo(tmp))
        trainer = get_trainer(cfg)
        trainer.prepare()
        calls = _capture(trainer)
        trainer.train()
        cmd = calls[0][0]
        assert cmd[cmd.index("-m") + 1] == "closd.diffusion_planner.train.train_mdm"
        assert "--data_dir" not in cmd and "--resume_checkpoint" not in cmd   # none set
        # with a checkpoint set, it's passed through (and existence-checked)
        ckpt = tmp / "c.pt"; ckpt.write_bytes(b"x")
        cfg2 = _cfg(tmp, method="closd", repo=_fake_repo(tmp), resume_checkpoint=ckpt)
        t2 = get_trainer(cfg2); t2.prepare()
        calls2 = _capture(t2); t2.train()
        assert _val(calls2[0][0], "--resume_checkpoint") == str(ckpt)


def test_real_trainers_require_repo_before_running():
    with tempfile.TemporaryDirectory() as tmp:
        for method in ("momask", "mdm", "protomotions", "closd"):
            cfg = _cfg(Path(tmp), method=method)
            assert_raises(TrainerError, lambda: get_trainer(cfg).train())


def test_staging_refuses_to_clobber_existing_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        repo = _fake_repo(tmp)
        (repo / "dataset").mkdir()
        (repo / "dataset" / "HumanML3D").mkdir()   # a real (non-symlink) dataset already there
        cfg = _cfg(tmp, method="momask", repo=repo)
        trainer = get_trainer(cfg); trainer.prepare()
        assert_raises(TrainerError, lambda: trainer.train())


# -- base helpers -----------------------------------------------------------

def test_run_cmd_error_paths():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = get_trainer(_cfg(Path(tmp)))
        assert_raises(TrainerError, lambda: trainer._run_cmd(["/no/such/interpreter"]))
        assert_raises(TrainerError, lambda: trainer._run_cmd([sys.executable, "-c", "import sys; sys.exit(3)"]))


def test_subprocess_env_pins_cuda_device():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp)); cfg.cuda_device = "1"
        trainer = get_trainer(cfg)
        assert trainer._subprocess_env()["CUDA_VISIBLE_DEVICES"] == "1"
        cfg.cuda_device = None
        assert trainer._subprocess_env() is None


def test_feature_formats_are_declared():
    fmts = {m: get_trainer(MotionModelConfig(method=m)).feature_format for m in available_methods()}
    assert fmts["momask"] == "humanml3d_263" and fmts["protomotions"] == "amass_smplh"
    assert fmts["noop"] == "synthetic"


# -- CLI --------------------------------------------------------------------

def test_cli_full_flow_with_noop():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds = _fake_dataset(tmp / "dataset")
        common = ["--dataset-dir", str(ds), "--work-root", str(tmp / "w"), "--method", "noop"]
        for sub in ("methods", "info", "prepare", "train"):
            assert cli.main([*common, sub]) == 0


def test_cli_discover_config_prefers_explicit():
    assert cli._discover_config("explicit.yaml") == "explicit.yaml"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} motion_model tests passed")


if __name__ == "__main__":
    _run_all()
