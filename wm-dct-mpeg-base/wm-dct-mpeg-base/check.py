#!/usr/bin/env python3
import pathlib
import re
import subprocess

def run(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)

video_ok = pathlib.Path("videos/watermarked.mpg").exists()
schedule_ok = pathlib.Path("out/key_schedule.json").exists()

extract = run(
    "python3 wm_tool.py extract "
    "--config wm_config.json "
    "--input videos/watermarked.mpg "
    "--schedule out/key_schedule.json "
    "--workdir out/check_extract"
)

txt = extract.stdout + extract.stderr

m = re.search(r"FINAL_BER=([0-9.]+)", txt)
ber = float(m.group(1)) if m else 1.0

detected = "DETECTED=1" in txt

psnr_proc = run(
    "python3 wm_tool.py psnr "
    "--main videos/watermarked.mpg "
    "--ref videos/source.mpg "
    "--stats out/check_psnr.log"
)

m = re.search(r"PSNR=([0-9.]+)", psnr_proc.stdout + psnr_proc.stderr)
psnr = float(m.group(1)) if m else 0.0

print(f"VIDEO_OK={int(video_ok)}")
print(f"SCHEDULE_OK={int(schedule_ok)}")
print(f"BER={ber:.4f}")
print(f"BER_OK={int(ber <= 0.10)}")
print(f"PSNR={psnr:.2f}")
print(f"PSNR_OK={int(psnr >= 33.0)}")
print(f"DETECTED={int(detected)}")
print(f"LAB_A_DONE={int(video_ok and schedule_ok and detected and ber <= 0.10 and psnr >= 33.0)}")
