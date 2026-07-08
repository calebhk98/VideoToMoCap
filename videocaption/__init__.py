"""VideoCaption (Pipeline 3): caption a personal video archive + index it for search.

Pipeline overview
-----------------
    raw video -> segment -> per-frame caption -> aggregate -> search index
    (up to 2h)   (Step 1)   (JoyCaption VLM)     (Dolphin)    (Step 5)
                                                                  |
                              group into windows (Step 5.5) ------+
                                                                  |
              LoRA fine-tune Qwen2.5-VL on the labels (Step 6, the deliverable)

Steps 1-5 exist to bootstrap the training data for Step 6: a video-native model
that captions a whole window in one pass. The heavy models (JoyCaption, Dolphin,
the Qwen fine-tune) run in their own environments behind subprocess bridges;
everything else here is plain standard library and runs without a GPU. See
``videocaption/README.md`` for the method research, licensing, and hardware.
"""

__version__ = "0.1.0"

from .config import CaptionConfig, load_config

__all__ = ["CaptionConfig", "load_config", "__version__"]
