# OpenEarthAgent Tool Server Model Weights

This guide lists the pretrained assets required by the tool server used for
[OpenEarth-Bench](https://github.com/mbzuai-oryx/OpenEarthAgent/tree/main).
In this project, the server is started with `scripts/tools/run_tools_oea.sh`.
The running tool server uses six model families. Download the files before
starting the services if the machine must run without internet access.

This document covers OpenEarthAgent assets only. Keep Earth-Bench model
instructions separately under `SkillEarth/models/eb/`.

## Environment Installation Order

Install the Conda environments in this order:

1. `vllm` for the OpenAI-compatible local LLM service.
2. `tool-server-e1` for the primary tool server and core GIS workers.
3. `tool-server-e2` for `ChangeDetection` and TEOChat.
4. `tool-server-e3` for `ObjectDetection` and LAE-DINO.

The `vllm` environment serves the planning model and is not a tool-server model
dependency. Install the following assets in the environment that runs their
worker:

| Environment | Tool assets in this guide |
| --- | --- |
| `tool-server-e1` | GroundingDINO, Qwen3-VL-8B-Instruct, SAM 2.1, EasyOCR, and the AddIndexLayer Earth Engine key |
| `tool-server-e2` | TEOChat for `ChangeDetection` |
| `tool-server-e3` | LAE-DINO for `ObjectDetection` |


Run every command below from the directory that contains `SkillEarth`. The
commands use paths relative to that location:

```text
SkillEarth/
├── models/
│   ├── oea/
│   │   ├── oea.md
│   │   ├── Qwen3-VL-8B-Instruct/
│   │   ├── OCR/
│   │   └── src/
│   │       ├── GroundingDINO/
│   │       ├── sam2/
│   │       ├── TEOChat/
│   │       ├── LAE-DINO/
│   │       ├── groundingdino_swint_ogc.pth
│   │       ├── sam2/checkpoints/sam2.1_hiera_large.pt
│   │       └── lae_dino_swint_lae1m-28ca3a15.pth
│   └── eb/
│       └── <Earth-Bench model documentation>
```

```bash
export PROJECT_ROOT=SkillEarth
export MODELS_DIR="$PROJECT_ROOT/models/oea"
mkdir -p "$MODELS_DIR/src" "$MODELS_DIR/OCR"
```

After downloading, update `$PROJECT_ROOT/tool_server/oea/tool_workers/scripts/launch_scripts/config/skillearth.yaml` to point to the corresponding paths under `$MODELS_DIR`.

## 1. GroundingDINO for `TextToBbox`

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git \
  "$MODELS_DIR/src/GroundingDINO"
pip install -e "$MODELS_DIR/src/GroundingDINO" --no-build-isolation

wget -q \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -O "$MODELS_DIR/src/groundingdino_swint_ogc.pth"
```

Configure `TextToBbox` with:

```text
model-path: <MODELS_DIR>/src/groundingdino_swint_ogc.pth
model-config: <MODELS_DIR>/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

## 2. Qwen3-VL-8B-Instruct for `BaseModel`

`BaseModel` is also used by `ImageDescription`,
`RegionAttributeDescription`, and `CountGivenObject`.

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir "$MODELS_DIR/Qwen3-VL-8B-Instruct"
```

Configure `BaseModel` with:

```text
model_path: <MODELS_DIR>/Qwen3-VL-8B-Instruct
```

## 3. SAM 2.1 Hiera Large for `SegmentObjectPixels`

Clone and install SAM2, then use its official checkpoint download script.
The script downloads several variants; this service uses
`sam2.1_hiera_large.pt`.

```bash
git clone https://github.com/facebookresearch/sam2.git "$MODELS_DIR/src/sam2"
pip install -e "$MODELS_DIR/src/sam2"

(
  cd "$MODELS_DIR/src/sam2/checkpoints"
  bash download_ckpts.sh
)
```

Configure `SegmentObjectPixels` with:

```text
sam_model_path: <MODELS_DIR>/src/sam2/checkpoints/sam2.1_hiera_large.pt
sam_model_config: configs/sam2.1/sam2.1_hiera_l.yaml
```

The configuration file is provided by the installed SAM2 package; it is not a
separate model weight.

## 4. TEOChat for `ChangeDetection`

The worker loads the Hugging Face model `jirvin16/TEOChat` from a local
directory.

```bash
huggingface-cli download jirvin16/TEOChat \
  --local-dir "$MODELS_DIR/src/TEOChat"
```

Configure `ChangeDetection` with:

```text
model_base: <MODELS_DIR>/src/TEOChat
```

## 5. LAE-DINO for `ObjectDetection`

The implementation requires both the LAE-DINO/MMDetection source tree and its
checkpoint. The source setup and Google Drive checkpoint are based on the
upstream OpenEarthAgent instructions.

```bash
git clone https://github.com/jaychempan/LAE-DINO.git "$MODELS_DIR/src/LAE-DINO"
pip install -e "$MODELS_DIR/src/LAE-DINO/mmdetection_lae" --no-build-isolation

gdown --fuzzy \
  'https://drive.google.com/uc?id=1EiR8KtNRYIeOfvtIe9C82cQk_uOMIQ8U' \
  -O "$MODELS_DIR/src/lae_dino_swint_lae1m-28ca3a15.pth"
```

Configure `ObjectDetection` with:

```text
model-path: <MODELS_DIR>/src/lae_dino_swint_lae1m-28ca3a15.pth
model-config: <MODELS_DIR>/src/LAE-DINO/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py
```

### Update the LAE-DINO import path

`ObjectDetection_worker.py` contains a source-level import path that is not
configured in the YAML file. After placing the LAE-DINO source tree, update
line 11 of
`tool_server/oea/tool_workers/online_workers/ObjectDetection_worker.py` so it
points to the installed MMDetection directory:

```python
sys.path.insert(0, "SkillEarth/models/oea/src/LAE-DINO/mmdetection_lae")
```

Replace `SkillEarth` if your project root uses a different directory name.

For fully offline startup, prefetch the BERT language model used by the
LAE-DINO configuration into the Hugging Face cache:

```bash
huggingface-cli download bert-base-uncased
```

## 6. EasyOCR weights for `OCR`

The worker creates an EasyOCR reader for `ch_sim` and `en` with
`download_enabled=False`. Pre-download the three required files into the
configured directory:

```bash
python - <<'PY'
import os
import easyocr

model_dir = os.environ["MODELS_DIR"] + "/OCR"
easyocr.Reader(
    ["ch_sim", "en"],
    model_storage_directory=model_dir,
    download_enabled=True,
)
PY
```

Verify that the directory contains:

```text
craft_mlt_25k.pth
english_g2.pth
zh_sim_g2.pth
```

### Update the OCR weight path

The download command above stores the weights in `SkillEarth/models/oea/OCR`.
Make `OCR_worker.py` use the same directory by updating line 70 of
`tool_server/oea/tool_workers/online_workers/OCR_worker.py`:

```python
model_storage_directory="SkillEarth/models/oea/OCR"
```

If you intentionally store OCR weights elsewhere, set this value to that
directory instead. The worker runs with `download_enabled=False`, so the path
must already contain all three files.

## AddIndexLayer

`AddIndexLayer` requires a Google Earth Engine (GEE) service-account JSON key.
Follow the official [Google Earth Engine service-account guide](https://developers.google.com/earth-engine/guides/service_account)
to create a service account, enable Earth Engine access, and download its JSON
key.

### Store the service-account JSON

Create `gee_key` at the SkillEarth project root and copy the downloaded JSON
file into it. Keep the original JSON contents unchanged.

```bash
mkdir -p SkillEarth/gee_key
cp <DOWNLOADED_SERVICE_ACCOUNT_KEY>.json \
  SkillEarth/gee_key/<service-account-key>.json
```

### Configure `skillearth.yaml`

Open
`SkillEarth/tool_server/oea/tool_workers/scripts/launch_scripts/config/skillearth.yaml`.
In the `AddIndexLayer` worker's `cmd` block, set the following two fields:

```yaml
service_account: "<service-account-name>@<gcp-project-id>.iam.gserviceaccount.com"
key_file: "SkillEarth/gee_key/<service-account-key>.json"
```

Set `service_account` to the `client_email` value in the downloaded JSON file.
Set `key_file` to the relative path of that same JSON file. Restart the tool
server after changing the configuration.
