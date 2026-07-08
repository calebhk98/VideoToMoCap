"""Command-line interface: ``python -m character <command>``.

Config-first, exactly like Pipelines 1 and 2: put every knob (method, repo paths,
prompt, critic, ...) in a YAML and the commands read it; flags are optional overrides.
Swapping the generation tool is one line (``method:``), so you can A/B the whole open
zoo -- IDOL, LHM, MakeHuman, SO-SMPL, ... -- from config alone.

    python -m character methods                       # list selectable tools
    python -m character --config c.yaml info          # what the configured tool will do
    python -m character --config c.yaml generate      # one character
    python -m character --config c.yaml refine        # generate -> critique mesh -> refine loop
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from . import available_methods, get_backend
from .config import CharacterConfig, load_config

_DEFAULT_CONFIG_NAMES = ("character.yaml", "character.yml", "configs/character.yaml")


def _discover_config(explicit) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get("CHARACTER_CONFIG")
    if env:
        return env
    for name in _DEFAULT_CONFIG_NAMES:
        if Path(name).exists():
            return name
    return None


def _cfg(args) -> CharacterConfig:
    path = _discover_config(args.config)
    if path:
        print(f"Using config: {path}")
    cfg = load_config(path) if path else CharacterConfig()
    for attr in ("method", "prompt", "work_root"):
        val = getattr(args, attr, None)
        if val:
            setattr(cfg, attr, Path(val) if attr == "work_root" else val)
    if getattr(args, "input_image", None):
        cfg.input_image = Path(args.input_image)
    return cfg


def cmd_methods(args) -> int:
    """List selectable tools, their input, and whether they drive with no retarget."""
    for name in available_methods():
        backend = get_backend(CharacterConfig(method=name))
        print(f"  {name:16s} {backend.describe()}")
    return 0


def cmd_info(args) -> int:
    cfg = _cfg(args)
    backend = get_backend(cfg)
    print(f"method:   {cfg.method}  ({backend.describe()})")
    print(f"prompt:   {cfg.prompt!r}")
    print(f"image:    {cfg.input_image}")
    print(f"asset:    {cfg.asset_dir}")
    print(f"critic:   {cfg.critic} (accept>={cfg.accept_score}, max_attempts={cfg.max_attempts})")
    return 0


def cmd_generate(args) -> int:
    """Generate one character with the configured tool."""
    cfg = _cfg(args)
    backend = get_backend(cfg)
    print(f"Generating character with method={cfg.method} ...")
    character = backend.generate()
    manifest = cfg.method_dir / "character.json"
    character.save_manifest(manifest)
    drive = "SMPL-X-native (drives with no retarget)" if character.native_smplx else f"rig={character.rig}"
    print(f"Character: {character.asset_path} [{drive}]")
    print(f"Manifest:  {manifest}")
    return 0


def cmd_refine(args) -> int:
    """Generate -> critique the MESH -> refine loop; keep the best candidate."""
    from . import loop
    from .critic import get_critic

    cfg = _cfg(args)
    backend = get_backend(cfg)
    critic = get_critic(cfg)
    print(f"Refining character (method={cfg.method}, critic={cfg.critic}) ...")
    result = loop.refine(cfg, backend, critic)
    print(loop.format_refine(result))
    if result.character is not None:
        result.character.save_manifest(cfg.method_dir / "character.json")
    return 0 if result.accepted else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="character", description=__doc__)
    p.add_argument("--config", help="path to character YAML config")
    p.add_argument("--method", help="override the configured tool")
    p.add_argument("--prompt", help="override the character description")
    p.add_argument("--input-image", dest="input_image", help="override the concept image")
    p.add_argument("--work-root", dest="work_root", help="override work/output root")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("methods", help="list selectable tools").set_defaults(func=cmd_methods)
    sub.add_parser("info", help="show what the configured tool will do").set_defaults(func=cmd_info)
    sub.add_parser("generate", help="generate one character").set_defaults(func=cmd_generate)
    sub.add_parser("refine", help="generate -> critique mesh -> refine loop").set_defaults(func=cmd_refine)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
