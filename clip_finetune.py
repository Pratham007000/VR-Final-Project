"""
clip_finetune.py — CLIP vision encoder fine-tuning with contrastive loss.

Fine-tunes the last N transformer blocks of CLIP's vision encoder
using triplet/contrastive loss so that images of the same item_id
are closer in embedding space.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip

import config
from utils import load_image, ensure_dir
from data_preparation import prepare_dataset, get_item_to_images

# ──────────────────────────────────────────────────────
# Dataset for contrastive training
# ──────────────────────────────────────────────────────

class TripletFashionDataset(Dataset):
    """
    Generates (anchor, positive, negative) triplets.
    Anchor and positive share the same item_id.
    Negative has a different item_id.
    """

    def __init__(self, item_to_images, preprocess, crop_dir=None, num_triplets=10000):
        self.preprocess = preprocess
        self.crop_dir = crop_dir
        self.items = list(item_to_images.keys())
        self.item_to_images = item_to_images

        # Only keep items with >= 2 images (need positive pairs)
        self.items = [it for it in self.items if len(item_to_images[it]) >= 2]
        self.num_triplets = num_triplets
        print(f"TripletDataset: {len(self.items)} items with >=2 images, {num_triplets} triplets")

    def __len__(self):
        return self.num_triplets

    def _load(self, path):
        if self.crop_dir:
            flat = path.replace("/", "_").replace("\\", "_")
            if not flat.lower().endswith((".jpg", ".png")):
                flat += ".jpg"
            cp = os.path.join(self.crop_dir, flat)
            if os.path.exists(cp):
                return self.preprocess(load_image(cp))
        return self.preprocess(load_image(path))

    def __getitem__(self, idx):
        anchor_item = random.choice(self.items)
        anchor_imgs = self.item_to_images[anchor_item]

        anchor_path, pos_path = random.sample(anchor_imgs, 2)

        neg_item = random.choice(self.items)
        while neg_item == anchor_item:
            neg_item = random.choice(self.items)
        neg_path = random.choice(self.item_to_images[neg_item])

        anchor = self._load(anchor_path)
        positive = self._load(pos_path)
        negative = self._load(neg_path)

        return anchor, positive, negative

# ──────────────────────────────────────────────────────
# Fine-tuning logic
# ──────────────────────────────────────────────────────

def freeze_clip_except_vision_last_n(model, n_blocks=4):
    for param in model.parameters():
        param.requires_grad = False

    visual = model.visual
    if hasattr(visual, "transformer"):
        blocks = visual.transformer.resblocks
    elif hasattr(visual, "trunk"):
        blocks = visual.trunk.blocks
    else:
        raise ValueError("Cannot find vision transformer blocks")

    total_blocks = len(blocks)
    unfreeze_from = max(0, total_blocks - n_blocks)
    print(f"Vision encoder: {total_blocks} blocks, unfreezing last {n_blocks} (from block {unfreeze_from})")

    for i, block in enumerate(blocks):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    if hasattr(visual, "ln_post"):
        for param in visual.ln_post.parameters():
            param.requires_grad = True
    if hasattr(visual, "proj") and visual.proj is not None:
        if isinstance(visual.proj, nn.Parameter):
            visual.proj.requires_grad = True
        else:
            for param in visual.proj.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()

def train_one_epoch(model, dataloader, optimizer, device, margin):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for anchor, positive, negative in tqdm(dataloader, desc="Training"):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        a_emb = F.normalize(model.encode_image(anchor), dim=-1)
        p_emb = F.normalize(model.encode_image(positive), dim=-1)
        n_emb = F.normalize(model.encode_image(negative), dim=-1)

        loss = triplet_loss(a_emb, p_emb, n_emb, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)

def finetune_clip(seed=42):
    print("=" * 60)
    print(f"CLIP Fine-Tuning (seed={seed})")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(config.DEVICE)

    freeze_clip_except_vision_last_n(model, n_blocks=config.CLIP_FT_UNFREEZE)

    data = prepare_dataset()
    train_df = data["train"]
    item_to_images = get_item_to_images(train_df)

    crop_dir = os.path.join(config.CROPS_DIR, "train")
    dataset = TripletFashionDataset(
        item_to_images, preprocess,
        crop_dir=crop_dir if os.path.exists(crop_dir) else None,
        num_triplets=len(train_df) * 2  
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.CLIP_FT_BATCH_SIZE,
        shuffle=True,
        num_workers=config.CLIP_FT_NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CLIP_FT_LR,
        weight_decay=0.01
    )

    ensure_dir(config.CLIP_FT_DIR)
    best_loss = float("inf")
    for epoch in range(config.CLIP_FT_EPOCHS):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer,
            config.DEVICE, config.CLIP_FT_MARGIN
        )
        print(f"Epoch {epoch+1}/{config.CLIP_FT_EPOCHS} — Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(config.CLIP_FT_DIR, f"clip_finetuned_seed{seed}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (loss={best_loss:.4f}) -> {save_path}")

    print(f"\nFine-tuning complete. Best loss: {best_loss:.4f}")
    return model

if __name__ == "__main__":
    for seed in config.RANDOM_SEEDS:
        finetune_clip(seed=seed)
