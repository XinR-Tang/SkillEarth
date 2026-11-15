# SkillEarth OEA Dataset

This directory contains the [OpenEarthAgent](https://github.com/mbzuai-oryx/OpenEarthAgent) (OEA) data required to reproduce SkillEarth skill extraction on the selected 1,050 training samples.

```text
SkillEarth/datasets/oea/
├── README.md
├── train_split.json
├── train_gpkgs/
│   ├── <sample_id>.gpkg
│   ├── <sample_id>.gpkg|layername=<layer_name>
│   ├── <sample_id>.gpkg|layername=<layer_name>.aux.xml
│   └── _cache_manifest.json
└── data/
    ├── README.md
    ├── gpkgs/
    ├── test/
    ├── train_part1/
    ├── train_part2/
    ├── test.json
    └── gpkgs.zip
```

## Contents

- `train_split.json`: the selected 1,050 OpenEarthAgent training trajectories used for SkillEarth skill extraction. Each record contains the task query, tool trajectory, labels, and referenced input images.
- `train_gpkgs/`: precomputed GeoPackage tool cache for the selected trajectories. It provides cached boundaries, POIs, and spectral-index observations so that skill extraction can reuse tool results without rerunning the corresponding online geospatial operations.
  - `<sample_id>.gpkg` is the GeoPackage cache associated with one sample.
  - `*|layername=*` and `*.aux.xml` are raster-layer and QGIS auxiliary files generated with the GeoPackages.
  - `_cache_manifest.json` records the cache construction metadata.
- `data/`: download location for the original [OpenEarthAgent dataset](https://huggingface.co/datasets/MBZUAI/OpenEarthAgent). Download it before running skill extraction; the image files referenced by `train_split.json` are stored in `train_part1/` and `train_part2/`. See [`data/README.md`](data/README.md) for the download command and complete setup instructions.

Before running skill extraction, edit `BasicConfig` and `ToolConfig` in `SkillEarth/skillearth/oea/config.py` as follows:

```python
train = "SkillEarth/datasets/oea/train_split.json"
image_dir = "SkillEarth/datasets/oea/data/train_part1,SkillEarth/datasets/oea/data/train_part2"
cache_dir = "SkillEarth/datasets/oea/train_gpkgs"
```
