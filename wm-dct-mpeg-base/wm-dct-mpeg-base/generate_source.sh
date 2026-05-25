#!/bin/bash
set -e

mkdir -p videos out

python3 wm_tool.py make-source \
  --out videos/source.mpg \
  --duration 16 \
  --width 352 \
  --height 288 \
  --rate 25 \
  --gop 12 \
  --q 4

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  videos/source.mpg > out/source_info.txt

echo "SOURCE_OK=1"
cat out/source_info.txt
