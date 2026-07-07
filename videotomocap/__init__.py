"""VideoToMoCap: turn hours of personal video into anonymized motion-capture data.

Pipeline overview
-----------------
    raw video  ->  ingest/exclude  ->  HMR backend  ->  pose/shape split  ->  dataset
    (10 cams)      (manifest)          (GVHMR/WHAM)     (drop identity)       (AMASS npz)

The heavy neural stages (HMR inference, motion-model training) shell out to the
upstream research tools; everything else in this package is plain NumPy and runs
without a GPU.  See ``README.md`` for the method comparison and rationale.
"""

__version__ = "0.1.0"

from .config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config", "__version__"]
