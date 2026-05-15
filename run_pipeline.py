"""
run_pipeline.py — Master script to orchestrate the full pipeline end-to-end.
"""

import argparse
import os

def step_prepare():
    print("\n" + "=" * 60 + "\nSTEP 1: Data Preparation\n" + "=" * 60)
    from data_preparation import prepare_dataset
    return prepare_dataset()

def step_crop():
    print("\n" + "=" * 60 + "\nSTEP 2: YOLO Cropping\n" + "=" * 60)
    from data_preparation import prepare_dataset
    from yolo_detector import crop_dataset
    crop_dataset(prepare_dataset(), save=True)

def step_caption():
    print("\n" + "=" * 60 + "\nSTEP 3: BLIP-2 Captioning\n" + "=" * 60)
    from data_preparation import prepare_dataset
    from blip2_captioner import caption_dataset
    caption_dataset(prepare_dataset())

def step_finetune():
    print("\n" + "=" * 60 + "\nSTEP 4: CLIP Fine-Tuning\n" + "=" * 60)
    from clip_finetune import finetune_clip
    import config
    for seed in config.RANDOM_SEEDS: finetune_clip(seed=seed)

def step_index():
    print("\n" + "=" * 60 + "\nSTEP 5: Build FAISS Indices\n" + "=" * 60)
    from index_builder import build_index_pipeline
    import config
    build_index_pipeline(alpha=1.0, suffix="_configA")
    for alpha in config.ALPHA_VALUES: build_index_pipeline(alpha=alpha, suffix="_configB")
    for seed in config.RANDOM_SEEDS:
        ft_path = os.path.join(config.CLIP_FT_DIR, f"clip_finetuned_seed{seed}.pt")
        if os.path.exists(ft_path):
            for alpha in config.ALPHA_VALUES: build_index_pipeline(alpha=alpha, finetuned_path=ft_path, suffix=f"_configC_seed{seed}")

def step_evaluate():
    print("\n" + "=" * 60 + "\nSTEP 6: Evaluation\n" + "=" * 60)
    from evaluate import run_ablation_study
    run_ablation_study()

def main():
    parser = argparse.ArgumentParser(description="Visual Product Search Pipeline")
    parser.add_argument("--step", choices=["prepare", "crop", "caption", "finetune", "index", "evaluate", "all"], default="all")
    args = parser.parse_args()

    steps = {"prepare": step_prepare, "crop": step_crop, "caption": step_caption, "finetune": step_finetune, "index": step_index, "evaluate": step_evaluate}
    
    if args.step == "all":
        for func in steps.values(): func()
        print("\n✅ All Pipeline Steps Executed Successfully!")
    else:
        steps[args.step]()

if __name__ == "__main__":
    main()
