#!/usr/bin/env bash
set -e

echo "=== Installing system packages ==="
apt-get update -qq
apt-get install -y -qq \
  tesseract-ocr \
  tesseract-ocr-ara \
  libgl1 \
  libglib2.0-0 \
  unrar \
  antiword

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing CPU torch + ultralytics for safety_world detector ==="
# CPU wheels only — Render starter has no GPU. ~320 MB additional.
# SAFETY_WORLD_WEIGHTS env var (set in Render dashboard) points the
# block at the committed data/models/safety_world_v1.pt -- a YOLO-Worldv2
# checkpoint with its prompt vocabulary reparameterized into the head at
# bake time. No CLIP needed at inference.
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.5.1" \
  "torchvision==0.20.1" \
  "ultralytics==8.4.75" \
  "opencv-python-headless==4.13.0.92"

echo "=== Build complete ==="
