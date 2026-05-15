"""
blip2_captioner.py — BLIP-2 captioning and Image-Text Matching module.
"""

import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import config
from utils import load_image, save_json, load_json

class BLIP2Captioner:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or config.BLIP2_MODEL
        self.device = device or config.DEVICE
        print(f"Loading BLIP-2: {self.model_name} on {self.device}")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu": self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_caption(self, image: Image.Image, prompt=None, max_length=None) -> str:
        prompt = prompt or config.CAPTION_PROMPT
        max_length = max_length or config.CAPTION_MAX_LENGTH
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, dtype=dtype)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_length, num_beams=3, early_stopping=True)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def caption_batch(self, image_paths, crop_dir=None, save_path=None):
        captions = load_json(save_path) if save_path and os.path.exists(save_path) else {}
        remaining = [p for p in image_paths if p not in captions]
        print(f"Generating captions for {len(remaining)} images...")
        for i, path in enumerate(tqdm(remaining, desc="BLIP-2 Captioning")):
            try:
                img = None
                if crop_dir:
                    flat = path.replace("/", "_").replace("\\", "_") + (".jpg" if not path.lower().endswith((".jpg", ".png")) else "")
                    cp = os.path.join(crop_dir, flat)
                    if os.path.exists(cp): img = load_image(cp)
                if img is None: img = load_image(path)
                captions[path] = self.generate_caption(img)
            except Exception as e:
                captions[path] = "clothing item"
            if save_path and (i + 1) % 100 == 0: save_json(captions, save_path)
        if save_path: save_json(captions, save_path)
        return captions

    @torch.no_grad()
    def compute_itm_score(self, image: Image.Image, text: str) -> float:
        """Computes semantic alignment score via negative log-likelihood loss."""
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device, dtype=dtype)
        outputs = self.model(**inputs, labels=inputs.get("input_ids"))
        return -outputs.loss.item()

    def rerank_candidates(self, query_image, candidates, captions):
        for cand in tqdm(candidates, desc="ITM Re-ranking"):
            caption = captions.get(cand["image_path"], "clothing item")
            try: cand["itm_score"] = self.compute_itm_score(query_image, caption)
            except Exception: cand["itm_score"] = -float("inf")
        return sorted(candidates, key=lambda c: c["itm_score"], reverse=True)

def caption_dataset(data):
    captioner = BLIP2Captioner()
    all_captions = {}
    for split in ["gallery", "train"]:
        if split in data:
            crop_dir = os.path.join(config.CROPS_DIR, split)
            save_path = os.path.join(config.CAPTIONS_DIR, f"captions_{split}.json")
            all_captions.update(captioner.caption_batch(data[split]["image_path"].tolist(), crop_dir=crop_dir, save_path=save_path))
    return all_captions

if __name__ == "__main__":
    from data_preparation import prepare_dataset
    caption_dataset(prepare_dataset())
