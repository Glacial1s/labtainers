#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np


# =========================
# Utility
# =========================

def run(cmd: List[str], quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=quiet)


def ensure_dir(path):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_dir(path):
    p = pathlib.Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_cfg(path: str) -> Dict:
    return json.loads(pathlib.Path(path).read_text())


def save_json(obj, path):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2))


# =========================
# Video IO
# =========================

def make_source(out_video: str, duration: int, width: int, height: int,
                rate: int, gop: int, q: int):
    ensure_dir(pathlib.Path(out_video).parent)
    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=duration={duration}:size={width}x{height}:rate={rate}",
        "-c:v", "mpeg2video",
        "-q:v", str(q),
        "-g", str(gop),
        "-bf", "2",
        "-pix_fmt", "yuv420p",
        out_video
    ])


def extract_frames(video: str, out_dir: str):
    clean_dir(out_dir)
    run([
        "ffmpeg", "-y",
        "-i", video,
        f"{out_dir}/frame_%05d.png"
    ], quiet=True)


def assemble_frames(frame_dir: str, out_video: str, rate: int,
                    gop, q=4, bitrate=None):
    ensure_dir(pathlib.Path(out_video).parent)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(rate),
        "-i", f"{frame_dir}/frame_%05d.png",
        "-c:v", "mpeg2video",
        "-g", str(gop),
        "-bf", "2",
        "-pix_fmt", "yuv420p"
    ]
    if bitrate:
        cmd += ["-b:v", bitrate]
    else:
        cmd += ["-q:v", str(q)]
    cmd += [out_video]
    run(cmd, quiet=True)


def psnr_video(main_video: str, ref_video: str, stats_file: str) -> float:
    ensure_dir(pathlib.Path(stats_file).parent)
    cmd = [
        "ffmpeg", "-i", main_video, "-i", ref_video,
        "-lavfi",
        f"[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[main];"
        f"[1:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[ref];"
        f"[main][ref]psnr=stats_file={stats_file}",
        "-f", "null", "-"
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    txt = p.stderr + p.stdout
    m = re.search(r"average:([0-9.]+)", txt)
    if not m:
        return 0.0
    return float(m.group(1))


# =========================
# Watermark bits and ECC
# =========================

def payload_bits(payload: str, nbits: int) -> np.ndarray:
    digest = hashlib.sha256(payload.encode()).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    if nbits > len(bits):
        raise ValueError("nbits too large for SHA-256 based payload bits.")
    return bits[:nbits].astype(np.uint8)


def repetition_encode(bits: np.ndarray, r: int) -> np.ndarray:
    if r <= 1:
        return bits.copy()
    return np.repeat(bits, r).astype(np.uint8)


def repetition_decode(bits: np.ndarray, r: int) -> np.ndarray:
    if r <= 1:
        return bits.copy()
    if len(bits) % r != 0:
        raise ValueError("Encoded bit length is not divisible by repetition factor.")
    chunks = bits.reshape((-1, r))
    return np.array([1 if np.sum(c) >= (r / 2.0) else 0 for c in chunks], dtype=np.uint8)


def ber(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return float(np.mean(a[:n] != b[:n]))


# =========================
# DCT embedding
# =========================

def block_variance(y: np.ndarray, by: int, bx: int) -> float:
    block = y[by*8:by*8+8, bx*8:bx*8+8]
    return float(np.var(block))


def block_edge(y: np.ndarray, by: int, bx: int) -> float:
    block = y[by*8:by*8+8, bx*8:bx*8+8].astype(np.float32)
    sx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(sx*sx + sy*sy)))


def adaptive_alpha(base_alpha: float, variance: float,
                   var_min: float, var_max: float) -> float:
    if var_max <= var_min:
        return base_alpha
    t = (variance - var_min) / (var_max - var_min)
    t = min(1.0, max(0.0, t))
    return base_alpha * (1.0 + 0.75 * t)


def embed_bit_in_block(block_y: np.ndarray, bit: int,
                       alpha: float,
                       coeff_a: Tuple[int, int],
                       coeff_b: Tuple[int, int]) -> np.ndarray:
    x = block_y.astype(np.float32) - 128.0
    d = cv2.dct(x)

    ar, ac = coeff_a
    br, bc = coeff_b

    a = float(d[ar, ac])
    b = float(d[br, bc])

    if bit == 1:
        diff = a - b
        if diff < alpha:
            delta = (alpha - diff) / 2.0
            d[ar, ac] += delta
            d[br, bc] -= delta
    else:
        diff = b - a
        if diff < alpha:
            delta = (alpha - diff) / 2.0
            d[br, bc] += delta
            d[ar, ac] -= delta

    y2 = cv2.idct(d) + 128.0
    return np.clip(y2, 0, 255).astype(np.uint8)


def detect_bit_from_block(block_y: np.ndarray,
                          coeff_a: Tuple[int, int],
                          coeff_b: Tuple[int, int]) -> Tuple[int, float]:
    x = block_y.astype(np.float32) - 128.0
    d = cv2.dct(x)

    ar, ac = coeff_a
    br, bc = coeff_b

    score = float(d[ar, ac] - d[br, bc])
    bit = 1 if score > 0 else 0
    confidence = abs(score)
    return bit, confidence


# =========================
# Schedule generation
# =========================

def candidate_positions(frames: List[pathlib.Path], cfg: Dict) -> List[Tuple[int, int, int, float]]:
    adaptive = bool(cfg.get("adaptive", False))
    iframe_stride = int(cfg.get("frame_stride", 1))

    var_min = float(cfg.get("var_min", 30))
    var_max = float(cfg.get("var_max", 1200))
    edge_min = float(cfg.get("edge_min", 0))

    candidates = []

    for fi, fp in enumerate(frames):
        if fi % iframe_stride != 0:
            continue

        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue

        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]

        h, w = y.shape
        by_max = h // 8
        bx_max = w // 8

        for by in range(1, by_max - 1):
            for bx in range(1, bx_max - 1):
                if adaptive:
                    v = block_variance(y, by, bx)
                    e = block_edge(y, by, bx)
                    if not (var_min <= v <= var_max and e >= edge_min):
                        continue
                    candidates.append((fi, by, bx, v))
                else:
                    candidates.append((fi, by, bx, 0.0))

    return candidates


def build_schedule(frames: List[pathlib.Path], cfg: Dict,
                   encoded_len: int) -> Dict:
    seed = int(cfg["seed"])
    embed_repeat = int(cfg.get("embed_repeat", 5))
    rng = np.random.default_rng(seed)

    candidates = candidate_positions(frames, cfg)
    need = encoded_len * embed_repeat

    if len(candidates) < need:
        raise RuntimeError(
            f"Not enough candidate blocks. Need {need}, have {len(candidates)}. "
            "Lower repeat, lower var_min/edge_min, or use larger/longer video."
        )

    chosen = rng.choice(len(candidates), size=need, replace=False)

    entries = []
    k = 0
    for symbol_idx in range(encoded_len):
        for _ in range(embed_repeat):
            fi, by, bx, var_value = candidates[int(chosen[k])]
            entries.append({
                "symbol_idx": symbol_idx,
                "frame": int(fi),
                "by": int(by),
                "bx": int(bx),
                "variance": float(var_value)
            })
            k += 1

    return {
        "version": 1,
        "entries": entries,
        "encoded_len": encoded_len,
        "embed_repeat": embed_repeat,
        "width": None,
        "height": None
    }


# =========================
# Main embed/extract
# =========================

def embed_video(cfg_path: str, input_video: str, output_video: str,
                schedule_out: str, workdir: str):
    cfg = load_cfg(cfg_path)

    rate = int(cfg.get("rate", 25))
    gop = int(cfg.get("gop", 12))
    q = int(cfg.get("mpeg_q", 4))

    src_frames_dir = f"{workdir}/src_frames"
    wm_frames_dir = f"{workdir}/wm_frames"

    extract_frames(input_video, src_frames_dir)
    clean_dir(wm_frames_dir)

    frames = sorted(pathlib.Path(src_frames_dir).glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames extracted.")

    nbits = int(cfg["nbits"])
    ecc_r = int(cfg.get("ecc_repetition", 1))

    base_bits = payload_bits(cfg["payload"], nbits)
    encoded_bits = repetition_encode(base_bits, ecc_r)

    schedule = build_schedule(frames, cfg, len(encoded_bits))

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    schedule["width"] = w
    schedule["height"] = h

    coeff_a = tuple(cfg.get("coeff_a", [3, 2]))
    coeff_b = tuple(cfg.get("coeff_b", [2, 3]))
    base_alpha = float(cfg.get("alpha", 14))
    adaptive = bool(cfg.get("adaptive", False))
    var_min = float(cfg.get("var_min", 30))
    var_max = float(cfg.get("var_max", 1200))

    by_frame: Dict[int, List[Dict]] = {}
    for e in schedule["entries"]:
        by_frame.setdefault(e["frame"], []).append(e)

    for fi, fp in enumerate(frames):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]

        for e in by_frame.get(fi, []):
            by = e["by"]
            bx = e["bx"]
            symbol_idx = e["symbol_idx"]
            bit = int(encoded_bits[symbol_idx])

            a = base_alpha
            if adaptive:
                a = adaptive_alpha(base_alpha, float(e["variance"]), var_min, var_max)

            block = y[by*8:by*8+8, bx*8:bx*8+8]
            y[by*8:by*8+8, bx*8:bx*8+8] = embed_bit_in_block(
                block, bit, a, coeff_a, coeff_b
            )

        ycc[:, :, 0] = y
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(f"{wm_frames_dir}/frame_{fi+1:05d}.png", out)

    assemble_frames(wm_frames_dir, output_video, rate=rate, gop=gop, q=q)
    save_json(schedule, schedule_out)

    print("EMBED_OK=1")
    print(f"OUTPUT_VIDEO={output_video}")
    print(f"SCHEDULE={schedule_out}")
    print(f"ENCODED_BITS={len(encoded_bits)}")
    print(f"EMBED_REPEAT={int(cfg.get('embed_repeat', 5))}")


def extract_video(cfg_path: str, input_video: str, schedule_path: str,
                  workdir: str):
    cfg = load_cfg(cfg_path)
    schedule = json.loads(pathlib.Path(schedule_path).read_text())

    tmp_frames = f"{workdir}/extract_frames"
    extract_frames(input_video, tmp_frames)

    frames = sorted(pathlib.Path(tmp_frames).glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames extracted for detection.")

    coeff_a = tuple(cfg.get("coeff_a", [3, 2]))
    coeff_b = tuple(cfg.get("coeff_b", [2, 3]))
    encoded_len = int(schedule["encoded_len"])
    ecc_r = int(cfg.get("ecc_repetition", 1))
    nbits = int(cfg["nbits"])
    search_radius = int(cfg.get("search_radius_blocks", 0))

    votes: List[List[int]] = [[] for _ in range(encoded_len)]
    confs: List[List[float]] = [[] for _ in range(encoded_len)]

    for e in schedule["entries"]:
        fi = int(e["frame"])
        by0 = int(e["by"])
        bx0 = int(e["bx"])
        symbol_idx = int(e["symbol_idx"])

        if fi >= len(frames):
            continue

        img = cv2.imread(str(frames[fi]), cv2.IMREAD_COLOR)
        if img is None:
            continue

        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]
        h, w = y.shape
        by_max = h // 8
        bx_max = w // 8

        best_bit = None
        best_conf = -1.0

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                by = by0 + dy
                bx = bx0 + dx
                if by < 0 or bx < 0 or by >= by_max or bx >= bx_max:
                    continue

                block = y[by*8:by*8+8, bx*8:bx*8+8]
                bit, conf = detect_bit_from_block(block, coeff_a, coeff_b)

                if conf > best_conf:
                    best_conf = conf
                    best_bit = bit

        if best_bit is not None:
            votes[symbol_idx].append(best_bit)
            confs[symbol_idx].append(best_conf)

    encoded_detected = []
    symbol_conf = []

    for i in range(encoded_len):
        if not votes[i]:
            encoded_detected.append(0)
            symbol_conf.append(0.0)
            continue

        s = sum(votes[i])
        bit = 1 if s >= (len(votes[i]) / 2.0) else 0
        encoded_detected.append(bit)
        symbol_conf.append(float(np.mean(confs[i])))

    encoded_detected = np.array(encoded_detected, dtype=np.uint8)
    decoded = repetition_decode(encoded_detected, ecc_r)
    expected = payload_bits(cfg["payload"], nbits)

    final_ber = ber(decoded, expected)
    raw_expected = repetition_encode(expected, ecc_r)
    raw_ber = ber(encoded_detected, raw_expected)

    avg_conf = float(np.mean(symbol_conf)) if symbol_conf else 0.0
    detected = int(final_ber <= float(cfg.get("detect_ber_threshold", 0.20)))

    print("EXTRACT_OK=1")
    print(f"RAW_BER={raw_ber:.4f}")
    print(f"FINAL_BER={final_ber:.4f}")
    print(f"CONFIDENCE={avg_conf:.4f}")
    print(f"DETECTED={detected}")


def attack_video(kind: str, input_video: str, output_video: str):
    ensure_dir(pathlib.Path(output_video).parent)

    if kind == "reencode_1200k":
        cmd = ["ffmpeg", "-y", "-i", input_video,
               "-c:v", "mpeg2video", "-b:v", "1200k", "-g", "12", "-bf", "2",
               "-pix_fmt", "yuv420p", output_video]
    elif kind == "reencode_600k":
        cmd = ["ffmpeg", "-y", "-i", input_video,
               "-c:v", "mpeg2video", "-b:v", "600k", "-g", "12", "-bf", "2",
               "-pix_fmt", "yuv420p", output_video]
    elif kind == "scale_down_up":
        cmd = ["ffmpeg", "-y", "-i", input_video,
               "-vf", "scale=264:216,scale=352:288",
               "-c:v", "mpeg2video", "-q:v", "5", "-g", "12", "-bf", "2",
               "-pix_fmt", "yuv420p", output_video]
    elif kind == "noise_light":
        cmd = ["ffmpeg", "-y", "-i", input_video,
               "-vf", "noise=alls=8:allf=t",
               "-c:v", "mpeg2video", "-q:v", "5", "-g", "12", "-bf", "2",
               "-pix_fmt", "yuv420p", output_video]
    elif kind == "crop_pad":
        cmd = ["ffmpeg", "-y", "-i", input_video,
               "-vf", "crop=336:272:8:8,pad=352:288:8:8",
               "-c:v", "mpeg2video", "-q:v", "5", "-g", "12", "-bf", "2",
               "-pix_fmt", "yuv420p", output_video]
    else:
        raise ValueError(f"Unknown attack kind: {kind}")

    run(cmd, quiet=True)
    print(f"ATTACK_CREATED={kind}")


def run_attack_suite(cfg_path: str, wm_video: str, schedule: str,
                     source_video: str, out_csv: str, workdir: str):
    attacks = [
        "reencode_1200k",
        "reencode_600k",
        "scale_down_up",
        "noise_light",
        "crop_pad"
    ]

    ensure_dir("attacks")
    ensure_dir(pathlib.Path(out_csv).parent)

    rows = []

    for a in attacks:
        outv = f"attacks/{a}.mpg"
        attack_video(a, wm_video, outv)

        # capture extraction output
        p = subprocess.run([
            sys.executable, __file__, "extract",
            "--config", cfg_path,
            "--input", outv,
            "--schedule", schedule,
            "--workdir", f"{workdir}/{a}"
        ], text=True, capture_output=True)

        txt = p.stdout + p.stderr
        m1 = re.search(r"RAW_BER=([0-9.]+)", txt)
        m2 = re.search(r"FINAL_BER=([0-9.]+)", txt)
        m3 = re.search(r"DETECTED=([01])", txt)

        raw = float(m1.group(1)) if m1 else 1.0
        final = float(m2.group(1)) if m2 else 1.0
        det = int(m3.group(1)) if m3 else 0

        psnr = psnr_video(outv, source_video, f"out/psnr_{a}.log")

        rows.append({
            "attack": a,
            "psnr": f"{psnr:.2f}",
            "ber_raw": f"{raw:.4f}",
            "ber_after_ecc": f"{final:.4f}",
            "detected": det
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "attack", "psnr", "ber_raw", "ber_after_ecc", "detected"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("ATTACK_SUITE_OK=1")
    print(f"CSV={out_csv}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-source")
    p.add_argument("--out", default="videos/source.mpg")
    p.add_argument("--duration", type=int, default=16)
    p.add_argument("--width", type=int, default=352)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--rate", type=int, default=25)
    p.add_argument("--gop", type=int, default=12)
    p.add_argument("--q", type=int, default=4)

    p = sub.add_parser("embed")
    p.add_argument("--config", default="wm_config.json")
    p.add_argument("--input", default="videos/source.mpg")
    p.add_argument("--output", default="videos/watermarked.mpg")
    p.add_argument("--schedule", default="out/key_schedule.json")
    p.add_argument("--workdir", default="out/work_embed")

    p = sub.add_parser("extract")
    p.add_argument("--config", default="wm_config.json")
    p.add_argument("--input", default="videos/watermarked.mpg")
    p.add_argument("--schedule", default="out/key_schedule.json")
    p.add_argument("--workdir", default="out/work_extract")

    p = sub.add_parser("psnr")
    p.add_argument("--main", default="videos/watermarked.mpg")
    p.add_argument("--ref", default="videos/source.mpg")
    p.add_argument("--stats", default="out/psnr.log")

    p = sub.add_parser("attack")
    p.add_argument("--kind", required=True)
    p.add_argument("--input", default="videos/watermarked.mpg")
    p.add_argument("--output", required=True)

    p = sub.add_parser("attack-suite")
    p.add_argument("--config", default="wm_config.json")
    p.add_argument("--input", default="videos/watermarked.mpg")
    p.add_argument("--schedule", default="out/key_schedule.json")
    p.add_argument("--source", default="videos/source.mpg")
    p.add_argument("--csv", default="out/robustness.csv")
    p.add_argument("--workdir", default="out/attack_extract")

    args = ap.parse_args()

    if args.cmd == "make-source":
        make_source(args.out, args.duration, args.width, args.height,
                    args.rate, args.gop, args.q)

    elif args.cmd == "embed":
        embed_video(args.config, args.input, args.output,
                    args.schedule, args.workdir)

    elif args.cmd == "extract":
        extract_video(args.config, args.input, args.schedule, args.workdir)

    elif args.cmd == "psnr":
        value = psnr_video(args.main, args.ref, args.stats)
        print(f"PSNR={value:.2f}")

    elif args.cmd == "attack":
        attack_video(args.kind, args.input, args.output)

    elif args.cmd == "attack-suite":
        run_attack_suite(args.config, args.input, args.schedule,
                         args.source, args.csv, args.workdir)


if __name__ == "__main__":
    main()
