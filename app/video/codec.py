"""Frames to a video file and back.

Writing uses the ffmpeg program (H.264, the codec nearly all shared video
uses), because the quality setting matters for this project: compression is the
main thing a video watermark has to survive. Reading uses OpenCV, so checking a
clip works even where ffmpeg is not installed.

H.264 in one paragraph, because it shapes everything in frames.py. A few frames
are stored whole ("keyframes"). The rest are stored as "this block moved from
there, plus a correction", and the correction is rounded off hard. Fine detail
that is small and does not move with the picture, which is exactly what an
invisible watermark is, gets rounded away first.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

MAX_FRAMES = 240       # about 8-10 seconds; keeps every step interactive
MAX_SIDE = 640
DEFAULT_CRF = 20       # ffmpeg's quality dial: 18 is near-transparent, 23 default, 35+ visibly rough


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def frame_types(data: bytes, limit: int = MAX_FRAMES) -> list[str] | None:
    """How the codec stored each frame, in display order: "I" whole picture
    (keyframe), "P" changes from an earlier frame, "B" changes from frames on
    both sides. Needs ffprobe, which ships with ffmpeg; None without it."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    fd, path = tempfile.mkstemp(suffix=".vid")
    os.close(fd)
    try:
        with open(path, "wb") as fh:
            fh.write(data)
        done = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type",
                               "-of", "csv=p=0", "-read_intervals", f"%+#{limit}", path],
                              capture_output=True, text=True, timeout=60)
        types = [line.strip().strip(",") for line in done.stdout.splitlines() if line.strip()]
        return types[:limit] if done.returncode == 0 and types else None
    except Exception:
        return None
    finally:
        os.unlink(path)


def encode(frames: list[np.ndarray], fps: float, crf: int = DEFAULT_CRF) -> bytes:
    """Compress uint8 RGB frames to an H.264 MP4 and return the file's bytes."""
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH; it is needed to write video.")
    h, w = frames[0].shape[:2]
    h, w = h // 2 * 2, w // 2 * 2                      # H.264 needs even sizes
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", path]
        raw = b"".join(np.ascontiguousarray(f[:h, :w]).tobytes() for f in frames)
        done = subprocess.run(cmd, input=raw, capture_output=True)
        if done.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + done.stderr.decode("utf-8", "replace")[-300:])
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def decode(data: bytes, max_frames: int = MAX_FRAMES, max_side: int = MAX_SIDE) -> dict:
    """Read a video file's frames as uint8 RGB. Long or large clips are cut
    to `max_frames` and scaled down to `max_side`, and the result says so."""
    fd, path = tempfile.mkstemp(suffix=".vid")
    os.close(fd)
    try:
        with open(path, "wb") as fh:
            fh.write(data)
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError("Not a readable video. MP4, MOV, WebM and MKV usually work.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        frames = []
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if max(rgb.shape[:2]) > max_side:
                f = max_side / max(rgb.shape[:2])
                rgb = cv2.resize(rgb, (round(rgb.shape[1] * f), round(rgb.shape[0] * f)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        cap.release()
    finally:
        os.unlink(path)
    if not frames:
        raise ValueError("The video has no frames that could be read.")
    if not 1 <= fps <= 240:
        fps = 24.0
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or "unknown"
    return {"frames": frames, "fps": float(fps), "width": width, "height": height, "codec": codec,
            "total_frames": max(total, len(frames)), "cut_short": total > len(frames),
            "scaled_down": max(width, height) > max_side}
