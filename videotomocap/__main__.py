"""Entry point for ``python -m videotomocap`` -- delegates to :mod:`videotomocap.cli`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
