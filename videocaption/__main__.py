"""Entry point for ``python -m videocaption`` -- delegates to :mod:`videocaption.cli`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
