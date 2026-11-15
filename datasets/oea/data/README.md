# OpenEarthAgent Data

Download the official [OpenEarthAgent dataset](https://huggingface.co/datasets/MBZUAI/OpenEarthAgent) and place its contents in this directory.

From the repository root, download the dataset with:

```bash
huggingface-cli download MBZUAI/OpenEarthAgent \
  --repo-type dataset \
  --local-dir SkillEarth/datasets/oea/data
```

The resulting directory layout should match the official `OpenEarthAgent/data` layout:

```text
SkillEarth/datasets/oea/data/
├── gpkgs/
├── test/
├── train_part1/
├── train_part2/
├── test.json
└── gpkgs.zip
```

If `gpkgs/` is not present after downloading, extract `gpkgs.zip` in this directory.

The `train_part1/` and `train_part2/` folders provide the training images referenced by the SkillEarth OEA data split. Configure their paths in `ToolConfig.image_dir` before running skill extraction.
