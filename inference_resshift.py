#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 4K SR enhancement using ResShift v3 (4-step).
# LQ images are already bicubic-upscaled to 4K; this script adds high-frequency
# details without downscaling, by disabling the internal bicubic upsampling step.

import os, sys
from PIL import Image
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
LQ_DIR     = SCRIPT_DIR.parent / 'data' / 'lq'
OUT_DIR    = SCRIPT_DIR.parent / 'data' / 'sr'

# ---------------------------------------------------------------------------
# Model weights (v3, 4-step real-SR)
# ---------------------------------------------------------------------------
CKPT_PATH  = SCRIPT_DIR / 'weights' / 'resshift_realsrx4_s4_v3.pth'
VQGAN_PATH = SCRIPT_DIR / 'weights' / 'autoencoder_vq_f4.pth'
CONFIG     = SCRIPT_DIR / 'configs' / 'realsr_swinunet_realesrgan256_journal.yaml'

# ---------------------------------------------------------------------------
# Patch settings for 4K images (already at target resolution, no upscaling)
# chop_size: patch side length in pixels drawn from the 4K image
# chop_stride: step between patch origins (overlap = chop_size - chop_stride)
# ---------------------------------------------------------------------------
CHOP_SIZE   = 512   # 512-pixel patches → 128×128 VQ latent (matches training depth)
CHOP_STRIDE = 448   # 64-pixel overlap on each edge for seamless blending


def build_sampler():
    configs = OmegaConf.load(str(CONFIG))
    configs.model.ckpt_path      = str(CKPT_PATH)
    configs.autoencoder.ckpt_path = str(VQGAN_PATH)
    configs.diffusion.params.sf  = 4

    assert CKPT_PATH.exists(),  f"Checkpoint not found: {CKPT_PATH}"
    assert VQGAN_PATH.exists(), f"VQ-GAN weights not found: {VQGAN_PATH}"

    lq_size = configs.model.params.get('lq_size', 64)

    sampler = ResShiftSampler(
        configs,
        sf=4,
        chop_size=CHOP_SIZE,
        chop_stride=CHOP_STRIDE,
        chop_bs=1,
        use_amp=True,
        seed=12345,
        padding_offset=lq_size,
        no_upscale=True,   # images already at 4K; skip internal bicubic upsampling
    )
    return sampler


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lq_images = sorted(LQ_DIR.glob('*.jpg'))
    if not lq_images:
        print(f"No .jpg images found in {LQ_DIR}")
        return

    print(f"Found {len(lq_images)} image(s) in {LQ_DIR}")
    print(f"Patch size: {CHOP_SIZE}px  stride: {CHOP_STRIDE}px  (no internal upscaling)")

    sampler = build_sampler()

    for lq_path in lq_images:
        stem    = lq_path.stem
        out_jpg = OUT_DIR / f"{stem}_out.jpg"
        tmp_png = OUT_DIR / f"{stem}.png"

        print(f"  {lq_path.name}  →  {out_jpg.name}")

        # inference() saves a PNG named <stem>.png into out_dir
        sampler.inference(
            str(lq_path),
            str(OUT_DIR),
            mask_path=None,
            bs=1,
            noise_repeat=False,
        )

        # Convert the intermediate PNG to *_out.jpg, preserving input EXIF
        if tmp_png.exists():
            # Read EXIF bytes from original LQ (includes orientation tag)
            src = Image.open(str(lq_path))
            exif_bytes = src.info.get('exif', b'')

            # Load SR result and save as JPEG with original EXIF
            sr_img = Image.open(str(tmp_png))
            save_kwargs = {'quality': 95, 'subsampling': 0}
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes
            sr_img.save(str(out_jpg), **save_kwargs)
            tmp_png.unlink()
        else:

            print(f"    WARNING: expected output {tmp_png} not found")

    print(f"\nDone. Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
