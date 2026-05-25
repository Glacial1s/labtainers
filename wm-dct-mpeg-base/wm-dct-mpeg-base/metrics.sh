#!/bin/bash
set -e

mkdir -p out

python3 wm_tool.py psnr \
  --main videos/watermarked.mpg \
  --ref videos/source.mpg \
  --stats out/psnr.log | tee out/psnr_summary.txt
