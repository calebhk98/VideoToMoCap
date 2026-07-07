"""Offline AMASS -> HumanML3D 263-d feature extraction (generator methods).

Turns the old manual notebook hand-off into an offline, scriptable step by
orchestrating VERIFIED upstream code -- we do not reimplement the feature math:

  FK:   poses[:, :66] + neutral body -> (T, 22, 3) joints, via ``human_body_prior``
        and the one-time registration-gated SMPL-H **neutral** model. No DMPL, no
        gender split -- betas are already zeroed, so neutral is both sufficient and
        correct for this privacy-preserving pipeline.
  263:  those joints -> Mathux/TMR's ``joints_to_guofeats`` (github.com/Mathux/TMR),
        which is byte-identical to official HumanML3D AND ships its own reference
        skeleton, so the gated AMASS clip 000021 is no longer needed.

Both steps need torch, so the whole thing runs as ONE subprocess in the generator
env (``trainer_python``) -- never inline, since the core env is torch-free. It runs
only when ``smpl_model`` + ``tmr_repo`` are set; otherwise the caller prints the
recipe. Not executable in the light test env, so verify one clip's features against
a reference HumanML3D output before a long training run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def has_assets(cfg) -> bool:
    """True when the neutral SMPL-H model and a cloned TMR repo are both present."""
    return bool(cfg.smpl_model and Path(cfg.smpl_model).exists()
                and cfg.tmr_repo and Path(cfg.tmr_repo).exists())


# One subprocess, run in the generator env: FK (human_body_prior) -> HumanML3D joint
# convention -> TMR's joints_to_guofeats -> 263-d .npy per clip. The lines mirror
# HumanML3D's raw_pose_processing (trans_matrix Y/Z swap) and TMR's compute step
# (the x-flip that turns that improper swap back into a proper rotation) verbatim.
_DRIVER = r"""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, {tmr!r})
from human_body_prior.body_model.body_model import BodyModel
from src.guofeats import joints_to_guofeats

TRANS = np.array([[1.0,0,0],[0,0,1.0],[0,1.0,0]])          # AMASS Z-up -> HumanML3D Y-up
bm = BodyModel(bm_fname={model!r}, num_betas=10)            # neutral; num_dmpls=None -> DMPL skipped

def fk(npz_path):
    d = np.load(npz_path)
    p, t = d["poses"], d["trans"]; n = len(t)
    with torch.no_grad():
        b = bm(root_orient=torch.Tensor(p[:, :3]), pose_body=torch.Tensor(p[:, 3:66]),
               pose_hand=torch.Tensor(p[:, 66:]), trans=torch.Tensor(t),
               betas=torch.zeros(n, 10))
    joints = np.dot(b.Jtr.detach().cpu().numpy(), TRANS)[:, :22]   # HumanML3D ./joints convention
    joints[..., 0] *= -1                                           # TMR: make the swap a proper rotation
    return joints

os.makedirs({vecs!r}, exist_ok=True)
for npz in sorted(glob.glob(os.path.join({amass!r}, "*.npz"))):
    feats = joints_to_guofeats(fk(npz))
    np.save(os.path.join({vecs!r}, os.path.splitext(os.path.basename(npz))[0] + ".npy"), feats)
print("wrote 263-d features to", {vecs!r})
"""


def _driver_script(cfg, vecs_dir: Path) -> str:
    """Render the subprocess driver for this config (kept small + inspectable)."""
    return _DRIVER.format(tmr=str(cfg.tmr_repo), model=str(cfg.smpl_model),
                          amass=str(cfg.amass_dir), vecs=str(vecs_dir))


def _run(cfg, script: str) -> None:
    """Run the driver in the generator env (has torch + human_body_prior + TMR deps)."""
    subprocess.run([cfg.trainer_python, "-c", script], check=True)


def extract(cfg, clips: List[dict], out: Path) -> Path:
    """AMASS npz -> 263-d new_joint_vecs, offline. Returns the vectors dir.

    Mean/Std are intentionally NOT recomputed: when fine-tuning you must use the
    pretrained checkpoint's shipped Mean.npy/Std.npy (its block-uniform norm), not
    corpus stats -- recompute only for a from-scratch run.
    """
    vecs_dir = out / "new_joint_vecs"
    _run(cfg, _driver_script(cfg, vecs_dir))
    return vecs_dir
