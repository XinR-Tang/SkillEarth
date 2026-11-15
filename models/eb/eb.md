# Earth-Bench Tool-Model Weights

This directory contains the local model code and checkpoints required by the
Earth-Bench tool server. Benchmark scene-data instructions are maintained
separately in `datasets/eb/README.md`.

## Download

The official Earth-Agent online-inference release provides a Baidu Netdisk
archive containing the `models/` folder required by its Earth-Bench tools:

- **Official Baidu Netdisk link:**
  <https://pan.baidu.com/s/1DPEHueuGLy9unuPUSqgQPQ?pwd=1234>
- **Extraction code:** `1234`
- **Official source:** [Earth-Agent online-inference README](https://github.com/opendatalab/Earth-Agent/blob/bdae0a4077d0e3e7f3ec4d87743ff2b7edf83649/README.MD#3-model-weights)

## Install the checkpoints

Extract the archive and merge the contents of its `models/` directory into
this directory, preserving the subdirectory names:

```text
SkillEarth/models/eb/
├── MSCN/
├── RemoteCLIP/
├── RemoteSAM/
├── SM3Det/
├── Strip-R-CNN/
├── InstructSAM/
└── sam2/
```

For an archive with a top-level `models/` directory:

```bash
cd /path/to/SkillEarth
rsync -a /path/to/extracted/models/ models/eb/
```

The command retains the source code already present in this repository and
adds or updates the corresponding checkpoints. Before starting the tool
server, confirm that the expected checkpoint files are present in the model
subdirectories above.
