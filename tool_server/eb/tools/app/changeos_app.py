# app.py
import os
import uuid
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
import changeos


app = FastAPI(title="ChangeOS Service")

OUTPUT_ROOT = Path(
    os.environ.get(
        "EARTH_AGENT_OUTPUT_DIR",
        Path(__file__).resolve().parents[2] / "tool_outputs",
    )
).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class ChangeOSRequest(BaseModel):
    pre_image_path: str
    post_image_path: str
    output_path: str = None


def resolve_output_path(output_path: str = None) -> Path:
    """Keep all generated files under the Earth-Bench tool output directory."""
    requested = Path(output_path) if output_path else Path(f"changeos_{uuid.uuid4().hex}.png")
    if requested.is_absolute():
        requested = Path(requested.name)

    resolved = (OUTPUT_ROOT / requested).resolve()
    if OUTPUT_ROOT not in resolved.parents:
        resolved = OUTPUT_ROOT / requested.name
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


# Initialize model
model = changeos.from_name('changeos_r101')


def process_image_to_1024(image):
    """Resize image to 1024x1024"""
    return resize(
        image,
        (1024, 1024),
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )


@app.post("/changeos/predict")
async def changeos_predict(req: ChangeOSRequest):
    """
    ChangeOS prediction endpoint.
    If pre_image_path == post_image_path: extract building footprints
    If pre_image_path != post_image_path: detect damage between two images
    """
    pre_path = req.pre_image_path
    post_path = req.post_image_path
    # Determine mode based on whether images are the same
    is_same_image = (pre_path == post_path)
    output_file = resolve_output_path(req.output_path)

    # Read images
    pre_image = imread(pre_path)
    post_image = imread(post_path)

    # Preprocess
    pre_image = process_image_to_1024(pre_image)
    post_image = process_image_to_1024(post_image)

    if is_same_image:
        # Building extraction mode
        loc, _ = model(pre_image, pre_image)
        result = np.uint8((loc != 0)) * 255
    else:
        # Damage detection mode
        _, dam = model(pre_image, post_image)
        result = np.uint8((dam == 4)) * 255

    imsave(str(output_file), result)

    return {
        "status": "success",
        "output_path": str(output_file),
        "mode": "building_extract" if is_same_image else "damage_detect"
    }
