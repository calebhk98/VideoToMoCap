"""Step 6 -- fine-tune the video-native model (the primary deliverable).

Steps 1-5 exist to bootstrap the training data for *this*: a LoRA fine-tune of a
video-native base (``Qwen2.5-VL-7B-Instruct``, Apache-2.0) that takes a whole
window in one pass and emits the full timestamped ``(start, end, description,
tags)`` list -- real dense video captioning, not one caption per tiny clip.

Two parts, mirroring ``motion_model``'s trainers:
  * :func:`build_dataset` -- pure: turn windows into training examples whose
    *target* is the exact Step 4 label format. Unit-tested, no heavy deps.
  * :class:`LoRAFinetune` -- shells out to an upstream trainer (LLaMA-Factory by
    default) for the LoRA run, then merges the adapter into the base weights so
    the distributed model is one self-contained repo.

Only the framework code is invoked here; the heavy trainer lives in its own env
and is only ever run as a subprocess. Licence: the fine-tune's *inputs* come from
Llama-3.1-derived models (JoyCaption/Dolphin), so if this model is distributed
publicly the Llama 3.1 Community Licence naming/notice terms apply -- see README.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import CaptionConfig
from .window import Window

# The instruction the fine-tuned model is trained to answer. Times in the target
# are relative to the window start (the model only ever sees the window).
INSTRUCTION = (
    "Watch the video and list every distinct event. Respond with a JSON array of "
    "objects, each with 'start' and 'end' (seconds from the start of this clip), a "
    "'description', and a list of 'tags'."
)


class FinetuneError(RuntimeError):
    """Raised for fine-tune setup/execution problems -- always with actionable context."""


def _window_target(window: Window) -> str:
    """The assistant target: the window's dense label list, times window-relative."""
    base = window.start
    labels = [
        {
            "start": round(e.start - base, 2),
            "end": round(e.end - base, 2),
            "description": e.description,
            "tags": list(e.tags),
        }
        for e in window.entries
    ]
    return json.dumps(labels, ensure_ascii=False)


def build_dataset(cfg: CaptionConfig, windows: List[Window]) -> Path:
    """Write the bootstrap fine-tune dataset (one example per window). Returns its path.

    Each record carries the source video, the window's ``[start, end]`` span, and a
    user/assistant message pair whose assistant turn is the Step 4 label format.
    Emitted as JSONL so it streams; the exact loader field names an upstream
    trainer wants (``videos``/``messages``) are set here for LLaMA-Factory's
    ShareGPT-style multimodal format.
    """
    out = cfg.finetune_dir / "dataset.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        for w in windows:
            record = {
                "video": w.rel_path,
                "window": [w.start, w.end],
                "messages": [
                    {"role": "user", "content": "<video>" + INSTRUCTION},
                    {"role": "assistant", "content": _window_target(w)},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return out


class LoRAFinetune:
    """Bridge to the upstream LoRA trainer; see module docstring."""

    name = "qwen2.5-vl-lora"

    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

    def _subprocess_env(self) -> Optional[dict]:
        device = self.cfg.cuda_device
        if device is None:
            return None
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        return env

    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
        try:
            subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                           check=True, env=self._subprocess_env())
        except FileNotFoundError as exc:
            raise FinetuneError(
                f"Could not launch the trainer: {exc}. Is `{self.cfg.finetune_python}` on PATH "
                f"and `finetune_repo` set to the cloned trainer (e.g. LLaMA-Factory)?"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise FinetuneError(f"trainer exited with status {exc.returncode} on: {printable}") from exc

    def _require_repo(self) -> Path:
        repo = self.cfg.finetune_repo
        if repo is None or not Path(repo).exists():
            raise FinetuneError(
                f"fine-tuning needs `finetune_repo` pointing at the cloned trainer "
                f"(got {repo!r}). See videocaption/README.md."
            )
        return Path(repo)

    def train_command(self) -> List[str]:
        """The LoRA training command (LLaMA-Factory's ``llamafactory-cli train``)."""
        dataset = self.cfg.finetune_dir / "dataset.jsonl"
        adapter = self.cfg.finetune_dir / "adapter"
        return [
            self.cfg.finetune_python, "-m", "llamafactory.cli", "train",
            "--stage", "sft",
            "--do_train", "true",
            "--model_name_or_path", self.cfg.finetune_base,
            "--dataset_dir", str(self.cfg.finetune_dir),
            "--dataset", str(dataset),
            "--finetuning_type", "lora",
            "--lora_rank", str(self.cfg.lora_rank),
            "--lora_alpha", str(self.cfg.lora_alpha),
            "--lora_target", "all",  # attention + MLP broadly, not a narrow adapter
            "--num_train_epochs", str(self.cfg.finetune_epochs),
            "--output_dir", str(adapter),
        ]

    def merge_command(self) -> List[str]:
        """The adapter-merge command -> one standalone checkpoint for distribution."""
        adapter = self.cfg.finetune_dir / "adapter"
        merged = self.cfg.finetune_dir / "merged"
        return [
            self.cfg.finetune_python, "-m", "llamafactory.cli", "export",
            "--model_name_or_path", self.cfg.finetune_base,
            "--adapter_name_or_path", str(adapter),
            "--finetuning_type", "lora",
            "--export_dir", str(merged),
        ]

    def run(self) -> Path:
        """LoRA-train, then (if ``merge_adapter``) merge. Returns the model dir."""
        repo = self._require_repo()
        if not (self.cfg.finetune_dir / "dataset.jsonl").exists():
            raise FinetuneError("no dataset.jsonl -- run `videocaption finetune-data` first.")
        self._run_cmd(self.train_command(), cwd=repo)
        if not self.cfg.merge_adapter:
            return self.cfg.finetune_dir / "adapter"
        self._run_cmd(self.merge_command(), cwd=repo)
        return self.cfg.finetune_dir / "merged"
