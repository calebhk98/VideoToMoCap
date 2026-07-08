"""Unit tests for Pipeline 3 (character): config, registry, noop backend, mesh
critics, and the generate->critique->refine loop.

GPU-free by construction -- the `noop` backend exercises the whole flow with no
torch/weights/repo, and the real backends' command construction is asserted by
capturing `_run_cmd` (so the upstream flags are actually pinned, not just "fails
without a repo"). Runs under pytest OR directly: `python tests/test_character.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from character import available_methods, get_backend
from character.base import BackendError, Character
from character.config import CharacterConfig, load_config
from character.critic import GeometricCritic, NoopCritic, get_critic
from character.loop import refine


def assert_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _cfg(tmp: Path, **kw) -> CharacterConfig:
    return CharacterConfig(work_root=tmp / "work", method=kw.pop("method", "noop"), **kw)


def _fake_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def _fake_image(tmp: Path) -> Path:
    img = tmp / "elf.png"
    img.write_bytes(b"PNG")
    return img


def _capture(backend) -> list:
    calls = []
    backend._run_cmd = lambda cmd, cwd=None: calls.append(([str(c) for c in cmd], cwd))
    return calls


# -- config + registry ------------------------------------------------------

def test_registry_lists_all_tools():
    for expected in ("noop", "lhm", "idol", "pshuman", "human3diffusion",
                     "icon", "econ", "sifu", "anigs", "humanorbit",
                     "en3d", "so_smpl", "mpfb2", "makehuman", "mblab"):
        assert expected in available_methods()


def test_gui_only_and_eol_tools_fail_loud_to_mpfb2():
    with tempfile.TemporaryDirectory() as tmp:
        for method in ("makehuman", "mblab"):
            try:
                get_backend(_cfg(Path(tmp), method=method)).generate()
                raise AssertionError(f"expected BackendError for {method}")
            except BackendError as exc:
                assert "mpfb2" in str(exc)


def test_so_smpl_is_text_native_and_requires_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), method="so_smpl", repo=_fake_repo(Path(tmp)), prompt="")
        assert get_backend(cfg).native_smplx        # SMPL-X native
        assert_raises(BackendError, lambda: get_backend(cfg).generate())   # empty prompt

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, method="so_smpl", repo=_fake_repo(tmp), prompt="a high elf")
        backend = get_backend(cfg)
        calls = _capture(backend)
        (cfg.repo / "outputs").mkdir(parents=True, exist_ok=True)
        (cfg.repo / "outputs" / "body.obj").write_bytes(b"obj")
        char = backend.generate()
        train = calls[0][0]
        assert "--train" in train and any("prompt=a high elf" in c for c in train)
        assert calls[1][0].count("--export") == 1 and char.native_smplx


def test_unknown_method_fails_loud():
    assert_raises(BackendError, lambda: get_backend(CharacterConfig(method="nope")))


def test_config_rejects_bad_enums():
    assert_raises(ValueError, lambda: CharacterConfig(critic="magic"))
    assert_raises(ValueError, lambda: CharacterConfig(renderer="unreal"))


def test_config_roundtrip_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("method: lhm\nprompt: a high elf\ncritic: geometric\n")
        cfg = load_config(p)
        assert cfg.method == "lhm" and cfg.prompt == "a high elf" and cfg.critic == "geometric"


def test_load_config_rejects_unknown_keys():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("method: noop\nnot_a_key: 1\n")
        assert_raises(ValueError, lambda: load_config(p))


# -- noop backend end-to-end ------------------------------------------------

def test_noop_generate_writes_smplx_native_character():
    with tempfile.TemporaryDirectory() as tmp:
        char = get_backend(_cfg(Path(tmp), prompt="a dwarf")).generate()
        assert char.asset_path.exists() and char.native_smplx and char.rig == "smplx"
        assert char.smplx_params is not None and char.smplx_params.exists()


def test_character_manifest_roundtrip():
    import json

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        char = get_backend(_cfg(tmp)).generate()
        m = tmp / "character.json"
        char.save_manifest(m)
        data = json.loads(m.read_text())
        assert data["native_smplx"] is True and data["rig"] == "smplx"


# -- mesh critics -----------------------------------------------------------

def _sphere_npz(path: Path, scale=(0.3, 1.0, 0.3)) -> Path:
    """A tall, symmetric point blob -> passes the geometric critic (human-ish)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= np.array(scale)
    np.savez(path, vertices=pts.astype(np.float32))
    return path


def test_geometric_critic_scores_human_proportion_and_symmetry():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        char = Character(method="x", prompt="", asset_path=_sphere_npz(Path(tmp) / "m.npz"))
        res = GeometricCritic(cfg).score(char, "a person")
        assert 0.0 <= res.score <= 1.0 and res.details["symmetry"] > 0.5


def test_geometric_critic_rejects_degenerate_mesh():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.npz"
        np.savez(bad, vertices=np.full((3, 3), np.nan, np.float32))
        res = GeometricCritic(_cfg(Path(tmp))).score(
            Character(method="x", prompt="", asset_path=bad), "a person")
        assert res.score == 0.0 and not res.passed


def test_get_critic_factory():
    assert NoopCritic(_cfg(Path("/tmp"))).name == "noop"
    assert get_critic(CharacterConfig(critic="geometric")).name == "geometric"


# -- refine loop ------------------------------------------------------------

def test_refine_accepts_when_critic_passes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), accept_score=0.8, max_attempts=5)
        backend = get_backend(cfg)
        critic = NoopCritic(cfg, scores=[0.3, 0.5, 0.9, 1.0])  # improves, crosses 0.8 on 3rd
        result = refine(cfg, backend, critic)
        assert result.accepted and result.best_score == 0.9
        assert len(result.attempts) == 3        # stopped as soon as it passed


def test_refine_returns_best_effort_when_never_passing():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), accept_score=0.99, max_attempts=3, n_candidates=1)
        result = refine(cfg, get_backend(cfg), NoopCritic(cfg, scores=[0.3, 0.7, 0.5]))
        assert not result.accepted and result.best_score == 0.7 and len(result.attempts) == 3


def test_refine_best_of_n_keeps_highest():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), accept_score=0.99, max_attempts=1, n_candidates=3)
        result = refine(cfg, get_backend(cfg), NoopCritic(cfg, scores=[0.2, 0.8, 0.4]))
        assert result.best_score == 0.8 and len(result.attempts) == 3


# -- real backends: command construction + guards ---------------------------

def test_lhm_command_and_smplx_native():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, method="lhm", repo=_fake_repo(tmp), input_image=_fake_image(tmp))
        backend = get_backend(cfg)
        assert backend.native_smplx
        calls = _capture(backend)
        # produce a fake .ply where the adapter looks, so generate() completes
        out = cfg.repo / "exps" / "meshs"
        out.mkdir(parents=True, exist_ok=True)
        (out / "elf.ply").write_bytes(b"PLY")
        char = backend.generate()
        cmd = calls[0][0]
        assert "LHM.launch" in cmd and any(c.startswith("image_input=") for c in cmd)
        assert char.asset_format == "gaussian_ply" and char.native_smplx


def test_econ_command_and_smplx_params():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _cfg(tmp, method="econ", repo=_fake_repo(tmp), input_image=_fake_image(tmp))
        backend = get_backend(cfg)
        calls = _capture(backend)
        objdir = cfg.asset_dir / "out" / "econ" / "obj"
        objdir.mkdir(parents=True, exist_ok=True)
        (objdir / "elf_full.obj").write_bytes(b"obj")
        (objdir / "elf_smpl.npy").write_bytes(b"npy")
        char = backend.generate()
        cmd = calls[0][0]
        assert "apps.infer" in cmd and "./configs/econ.yaml" in cmd
        assert char.rig == "smplx" and char.smplx_params.name.endswith("_smpl.npy")


def test_image_native_backend_requires_image():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp), method="lhm", repo=_fake_repo(Path(tmp)))  # no input_image
        assert_raises(BackendError, lambda: get_backend(cfg).generate())


def test_blocked_backend_fails_loud_with_alternative():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            get_backend(_cfg(Path(tmp), method="anigs")).generate()
            raise AssertionError("expected BackendError")
        except BackendError as exc:
            assert "lhm" in str(exc).lower()


def test_real_backends_require_repo():
    with tempfile.TemporaryDirectory() as tmp:
        for method in ("lhm", "idol", "pshuman", "icon", "econ", "sifu"):
            cfg = _cfg(Path(tmp), method=method, input_image=_fake_image(Path(tmp)))
            assert_raises(BackendError, lambda: get_backend(cfg).generate())


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} character tests passed")


if __name__ == "__main__":
    _run_all()
