# dropzone/ — drop your videos here

This is the single folder to dump footage into. Then run the pipeline against it.

## How to use

1. Copy (or symlink) your videos into this folder. Two layouts both work:

   **Per-camera (recommended)** — the pipeline uses the top folder name as the
   camera id, which lets you mark specific cameras as static:
   ```
   dropzone/
     cam01_corner/2024-05-01/0800.mp4
     cam03_eyelevel/2024-05-01/1200.mp4
     cam07_corner/2025-01-02/1500.mp4
   ```

   **Flat** — just dump files; every clip gets the camera id `cam`:
   ```
   dropzone/
     clip_0001.mp4
     clip_0002.mp4
   ```

2. Run the pipeline pointed at this folder (config `configs/dropzone.yaml`
   already sets `footage_root: dropzone`):
   ```bash
   python -m videotomocap --config configs/dropzone.yaml scan
   python -m videotomocap --config configs/dropzone.yaml exclude --pattern "*/2024-12-24/*"
   python -m videotomocap --config configs/dropzone.yaml run
   ```

Nothing here is copied or transcoded — the pipeline reads the files in place and
writes all outputs to `work/`. Symlinks are fine, so you can point at your
existing 2-year archive without duplicating terabytes:
```bash
ln -s /mnt/archive/cam07 dropzone/cam07_corner
```

## Notes

- Supported extensions are set in the config (`video_exts`): `.mp4 .mov .mkv ...`
- The media you drop here is **git-ignored** — only this README and `.gitkeep`
  are tracked. Never commit footage.
- Discrete pan cameras: if a camera snaps between fixed presets, put each preset
  in its own subfolder so they're treated as independent clips.
