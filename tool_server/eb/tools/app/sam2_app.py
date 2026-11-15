# env: fastapi
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_ROOT = Path(__file__).resolve().parents[4] / "models" / "eb"
OUTPUT_ROOT = Path(
    os.environ.get(
        "EARTH_AGENT_OUTPUT_DIR",
        Path(__file__).resolve().parents[2] / "tool_outputs",
    )
).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL_ROOT / "sam2"))

import cv2
import asyncio
import torch
import numpy as np
import uuid
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


app = FastAPI(title="SAM2 Service")


class Sam2Request(BaseModel):
    image_path: str
    bbox: Optional[list] = None
    bboxes: Optional[list] = None
    output_path: Optional[str] = None


def resolve_output_path(output_path: Optional[str], default_name: str) -> Path:
    """Keep all generated files under the Earth-Bench tool output directory."""
    requested = Path(output_path) if output_path else Path(default_name)
    if requested.is_absolute():
        requested = Path(requested.name)

    resolved = (OUTPUT_ROOT / requested).resolve()
    if OUTPUT_ROOT not in resolved.parents:
        resolved = OUTPUT_ROOT / requested.name
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


class SAM2Engine:
    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        max_concurrency: int = 1,
        output_dir: str = str(OUTPUT_ROOT)
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam2(model_cfg, checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _apply_color_map(self, masks):
        if len(masks) == 0:
            return None

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        h, w = masks.shape[-2:]
        segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(len(masks)):
            mask = masks[i]
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            segmentation_map[mask > 0] = color

        return segmentation_map

    @staticmethod
    def _normalize_boxes(bbox, bboxes):
        """Return a list of [x1, y1, x2, y2] boxes from either parameter."""
        if bboxes:
            return [list(box) for box in bboxes]
        if bbox is None:
            return None
        # A single flat box: [x1, y1, x2, y2].
        if (
            len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            return [list(bbox)]
        # Otherwise bbox is already a (possibly nested) list of boxes.
        return [list(box) for box in bbox]

    def _infer(self, image_path: str, bbox: list = None, bboxes: list = None,
               output_path: str = None):
        # Load image from file path
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": f"Failed to read image: {image_path}"}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box_list = self._normalize_boxes(bbox, bboxes)
        if not box_list:
            return {"status": "error", "message": "SAM2 requires at least one bbox or bboxes prompt"}

        with torch.inference_mode():
            self.predictor.set_image(image)

            # SAM2 accepts an array of boxes (N, 4) and returns one mask per
            # box, so a multi-box request produces a single combined,
            # color-coded segmentation map.
            input_boxes = np.array(box_list, dtype=np.float32)
            masks, scores, logits = self.predictor.predict(
                box=input_boxes,
                multimask_output=False
            )

        seg_map = self._apply_color_map(masks)

        if seg_map is not None:
            file_name = f"seg_{uuid.uuid4().hex}.png"
            file_path = resolve_output_path(output_path, file_name)
            Image.fromarray(seg_map).save(file_path)

            return {
                "status": "success",
                "mask_path": str(file_path),
                "scores": scores.tolist() if hasattr(scores, 'tolist') else scores
            }

        return {"status": "error", "message": "No masks generated"}

    async def infer(self, image_path: str, bbox: list = None, bboxes: list = None,
                    output_path: str = None):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path, bbox, bboxes, output_path
            )


# Initialize engine
CHECKPOINT = str(MODEL_ROOT / "sam2/checkpoints/sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

engine = SAM2Engine(
    checkpoint=CHECKPOINT,
    model_cfg=MODEL_CFG,
    max_concurrency=1
)


@app.post("/sam2/predict")
async def sam2_predict(req: Sam2Request):
    try:
        return await engine.infer(
            req.image_path, req.bbox, req.bboxes, req.output_path
        )
    except Exception as exc:
        # Return a readable error instead of an opaque HTTP 500 so the agent
        # can report or repair rather than retry blindly.
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}"
        }
