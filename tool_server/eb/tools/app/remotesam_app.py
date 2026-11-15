# env: RemoteSAM
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_ROOT = Path(__file__).resolve().parents[4] / "models" / "eb"
sys.path.insert(0, str(MODEL_ROOT))

import cv2
import asyncio
import logging
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from RemoteSAM.tasks.code.model import RemoteSAM, init_demo_model


app = FastAPI(title="RemoteSAM Service")
logger = logging.getLogger(__name__)


class RemoteSAMRequest(BaseModel):
    image_path: str
    query_text: str


class RemoteSAMEngine:
    def __init__(
        self,
        checkpoint: str,
        max_concurrency: int = 1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_model = init_demo_model(checkpoint, self.device)

        self.model = RemoteSAM(base_model, self.device, use_EPOC=True)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _infer(self, image_path: str, query_text: str):
        try:
            # Load image from file path
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "message": f"Failed to read image: {image_path}"}
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                bbox = self.model.visual_grounding(image=image, sentence=query_text)

                bbox = [float(x) for x in bbox]

                return {
                    "status": "success",
                    "bbox": bbox
                }
        except Exception as exc:
            logger.exception("RemoteSAM inference failed")
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}

    async def infer(self, image_path: str, query_text: str):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._infer, image_path, query_text
            )


# Initialize engine
CHECKPOINT = str(MODEL_ROOT / "RemoteSAM/checkpoint/RemoteSAMv1.pth")

engine = RemoteSAMEngine(
    checkpoint=CHECKPOINT,
    max_concurrency=1
)


@app.post("/remotesam/predict")
async def remotesam_predict(req: RemoteSAMRequest):
    output = await engine.infer(req.image_path, req.query_text)

    return output
