"""
yolo_detector.py — YOLO-based product localization module.

Detects the primary clothing item in an image and returns a crop.
Supports body-region selection: upper body, lower body, or full body.
Uses YOLOv8 (pre-trained, frozen).
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

import config
from utils import load_image, ensure_dir

# ──────────────────────────────────────────────
# Body region definitions
# ──────────────────────────────────────────────
PERSON_CLASS_ID = 0

UPPER_BODY_RATIO = (0.0, 0.55)   # top 55% of person bbox
LOWER_BODY_RATIO = (0.40, 1.0)   # bottom 60% of person bbox
FULL_BODY_RATIO  = (0.0, 1.0)    # entire person bbox

REGION_RATIOS = {
    "upper_body": UPPER_BODY_RATIO,
    "lower_body": LOWER_BODY_RATIO,
    "full_body":  FULL_BODY_RATIO,
}

class YOLODetector:
    """Wraps YOLOv8 for clothing item detection and cropping."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.YOLO_MODEL
        self.device = device or config.DEVICE
        print(f"Loading YOLO model: {self.model_name} on {self.device}")
        self.model = YOLO(self.model_name)
    
    def detect(self, image_path: str, conf_threshold: float = 0.25):
        results = self.model(image_path, verbose=False, device=self.device)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu())
                    cls_id = int(box.cls[0].cpu())
                    if conf >= conf_threshold:
                        detections.append((x1, y1, x2, y2, conf, cls_id))
        return detections

    def _get_person_bbox(self, image_path: str, detections=None):
        if detections is None:
            detections = self.detect(image_path)
        person_dets = [d for d in detections if d[5] == PERSON_CLASS_ID]
        if person_dets:
            return max(person_dets, key=lambda d: d[4])
        elif detections:
            return max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        else:
            return None

    def get_body_region_crops(self, image_path: str, bbox_fallback: tuple = None, padding_ratio: float = 0.05) -> dict:
        img = load_image(image_path)
        w, h = img.size

        detections = self.detect(image_path)
        person_det = self._get_person_bbox(image_path, detections)

        if person_det is not None:
            px1, py1, px2, py2 = person_det[:4]
        elif bbox_fallback is not None:
            px1, py1, px2, py2 = bbox_fallback
        else:
            px1, py1, px2, py2 = 0, 0, w, h

        person_h, person_w = py2 - py1, px2 - px1
        crops = {}
        for region_name, (top_ratio, bot_ratio) in REGION_RATIOS.items():
            ry1 = py1 + person_h * top_ratio
            ry2 = py1 + person_h * bot_ratio
            rx1, rx2 = px1, px2

            pad_x = person_w * padding_ratio
            pad_y = (ry2 - ry1) * padding_ratio
            rx1, ry1 = max(0, rx1 - pad_x), max(0, ry1 - pad_y)
            rx2, ry2 = min(w, rx2 + pad_x), min(h, ry2 + pad_y)

            crops[region_name] = img.crop((int(rx1), int(ry1), int(rx2), int(ry2)))

        return {"crops": crops, "detections": detections, "person_bbox": (px1, py1, px2, py2)}

    def crop_main_item(self, image_path: str, bbox_fallback: tuple = None, padding_ratio: float = 0.05, region: str = "full_body") -> Image.Image:
        result = self.get_body_region_crops(image_path, bbox_fallback=bbox_fallback, padding_ratio=padding_ratio)
        return result["crops"].get(region, result["crops"]["full_body"])
    
    def crop_batch(self, image_paths: list, bboxes: dict = None, save_dir: str = None, region: str = "full_body") -> dict:
        if save_dir: ensure_dir(save_dir)
        results = {}
        for path in tqdm(image_paths, desc=f"YOLO Cropping ({region})"):
            if not os.path.exists(path): continue
            bbox_fb = bboxes.get(path) if bboxes else None
            try:
                cropped = self.crop_main_item(path, bbox_fallback=bbox_fb, region=region)
                if save_dir:
                    flat_name = path.replace("/", "_").replace("\\", "_")
                    if not flat_name.lower().endswith((".jpg", ".png")): flat_name += ".jpg"
                    save_path = os.path.join(save_dir, flat_name)
                    cropped.save(save_path)
                    results[path] = save_path
                else:
                    results[path] = cropped
            except Exception:
                try: results[path] = load_image(path)
                except Exception: pass
        return results

def crop_dataset(data: dict, save: bool = True) -> dict:
    detector = YOLODetector()
    all_crops = {}
    for split_name, df in data.items():
        print(f"\nCropping {split_name} split ({len(df)} images)...")
        image_paths = df["image_path"].tolist()
        bboxes = {row["image_path"]: (row["x1"], row["y1"], row["x2"], row["y2"]) for _, row in df.iterrows() if not pd.isna(row.get("x1"))} if "x1" in df.columns else None
        save_dir = os.path.join(config.CROPS_DIR, split_name) if save else None
        all_crops.update(detector.crop_batch(image_paths, bboxes=bboxes, save_dir=save_dir))
    return all_crops

if __name__ == "__main__":
    from data_preparation import prepare_dataset
    crop_dataset(prepare_dataset(), save=True)
