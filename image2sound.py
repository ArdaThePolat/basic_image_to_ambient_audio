#!/usr/bin/env python3
"""
Image → ambient soundscape (MVP)
- Image is captioned with BLIP
- Caption is used to prompt Stable Audio Open (Diffusers)
- Writes a WAV file (float32, stereo, 44.1 kHz)

Usage:
  python image2sound.py --image beach.jpg --out soundscape.wav --seconds 20 --seed 42
Auth:
  - Accept the model license on the HF model page beforehand.
  - Provide a token via env HUGGINGFACE_TOKEN or --hf-token.
"""

import os
import argparse
import numpy as np
import torch
import soundfile as sf
from PIL import Image
from huggingface_hub import login as hf_login
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableAudioPipeline

def login_if_needed(token: str | None):
    token = token or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        try:
            hf_login(token.strip(), add_to_git_credential=False)
        except Exception as e:
            print(f"[warn] Hugging Face login failed: {e}")

def caption_image(img_path: str, device: str = "cpu", max_new_tokens: int = 30) -> str:
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap  = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    img = Image.open(img_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)
    ids = cap.generate(**inputs, max_new_tokens=max_new_tokens)
    return proc.decode(ids[0], skip_special_tokens=True)

def build_prompt(caption: str) -> tuple[str, str]:
    prompt = (
        f"High-quality stereo ambient field recording of {caption}. "
        "No music. Smooth wide scene, gentle dynamics, realistic reverb."
    )
    negative = "music, melody, clipping, distortion, artifacts"
    return prompt, negative

def ensure_audio_writeable(x: np.ndarray) -> np.ndarray:
    # [N, C] float32 in [-1, 1], finite, contiguous
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.ndim == 2 and x.shape[0] == 2 and x.shape[1] > 2:
        x = x.T  # (2, N) -> (N, 2)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(x)

def generate_audio(prompt: str, negative: str, seconds: float, device: str, seed: int | None):
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    out = pipe(
        prompt,
        negative_prompt=negative,
        num_inference_steps=200,
        audio_end_in_s=float(seconds),
        num_waveforms_per_prompt=1,
        generator=gen,
    ).audios[0]

    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()

    return out, 44100  # SAO sample rate

def main():
    ap = argparse.ArgumentParser(description="Photo → ambient soundscape")
    ap.add_argument("--image", "-i", required=True, help="Path to input image (e.g., beach.jpg)")
    ap.add_argument("--out", "-o", default="soundscape.wav", help="Output WAV path")
    ap.add_argument("--seconds", "-s", type=float, default=20.0, help="Audio length in seconds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cpu", "cuda"], help="Inference device")
    ap.add_argument("--hf-token", default=None, help="Hugging Face token (or set HUGGINGFACE_TOKEN)")
    args = ap.parse_args()
    
    print("Using device:", args.device)

    login_if_needed(args.hf_token)

    caption = caption_image(args.image, device=args.device)
    print(f"[info] Caption: {caption}")

    prompt, negative = build_prompt(caption)
    audio, sr = generate_audio(prompt, negative, args.seconds, args.device, args.seed)

    audio = ensure_audio_writeable(audio)
    sf.write(args.out, audio, int(sr), format="WAV", subtype="FLOAT")
    print(f"[ok] Wrote {args.out} ({audio.shape[0]/sr:.1f}s @ {sr} Hz)")

if __name__ == "__main__":
    main()