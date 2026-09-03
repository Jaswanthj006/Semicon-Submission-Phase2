#!/usr/bin/env python3
"""Phase 2 Drift-Sense blind dataset generator.

Extends the Phase 1 generator to match the Phase 2 addendum exactly:

  - Zoom ratio: unknown, uniform in [8, 12]x   (Phase 1 was fixed 10x)
  - Rotation:   unknown, +/-5 deg, reported     (Phase 1 was noise-only, 1-3 deg)
  - ~20% of pairs have NO true instance (Set C) (Phase 1 always had a match)
  - Required per-pair fields: x, y, theta, scale, found, score-ready ground truth
  - Set D optical bonus: real 3-channel RGB, reference present

Produces the 200-pair blind structure from the addendum:
    Set A  70 pairs  nominal pose       (grayscale, reference present)
    Set B  70 pairs  degraded           (grayscale, reference present)
    Set C  40 pairs  absent             (grayscale, no true instance -> found=0)
    Set D  20 pairs  optical (bonus)    (RGB, reference present)
Sets A+B+C = 180 grayscale pairs feed rejection F1 / localization / pose.
Set D is bonus-only and never mixed into the grayscale F1 pool.

Two files come out, deliberately separate (mirrors how organizers would
actually hold out ground truth from contestants):
    pairs.csv         id, set, reference_path, search_path   (what a team gets)
    ground_truth.csv  id, set, found, gt_x, gt_y, gt_theta, gt_scale

    python generate_dataset_phase2.py --output-dir ./dataset_phase2 --seed 7

Dependencies: numpy, opencv-python (no torch).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

# =============================================================================
# Geometry
# =============================================================================
REFERENCE_SIZE_PX = 1000
SEARCH_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1

# Phase 2 disclosed bounds (addendum, slide 31 / 32). Hard-coding these is
# explicitly allowed -- "the disclosed ranges are exact; hard-coding the
# search bounds is intended, not a loophole."
SCALE_RATIO_RANGE = (8.0, 12.0)
ROTATION_DEG_RANGE = (-5.0, 5.0)
ABSENT_FRACTION = 0.20  # ~20% of pairs contain no true instance (Set C)
# Four monotonic severity steps for Set B — harsher than organizer calibration band.
DEGRADATION_LEVELS = (1.05, 1.25, 1.50, 1.80)

# Label verification gate (spec section 5). A label no correct matcher can
# reproduce is a broken label, so every present pair is template-matched
# against the *shipped* PNGs before it is accepted.
VERIFY_MAX_ERR_PX = 3.0        # global peak must land this close to the label
VERIFY_MARGIN_FLOOR = 0.02     # hard floor: below this the pair is a coin flip
VERIFY_MARGIN_PREFER = 0.12    # resample until this is cleared, then fall back
VERIFY_CROP_ATTEMPTS = 8       # crop resamples per canvas (search image reused)
VERIFY_CANVAS_ATTEMPTS = 3     # full canvas regenerations before failing loudly

# =============================================================================
# DRAM 6F2 / FinFET presets (public pitch ratios, not fab numbers)
# =============================================================================
DRAM_1X = {
    "kind": "dram", "feature_size_nm": 32,
    "word_line_pitch_nm": 64, "word_line_width_nm": 32,
    "bit_line_pitch_nm": 96, "bit_line_width_nm": 32, "contact_diameter_nm": 32,
}
DRAM_DENSE = {
    "kind": "dram", "feature_size_nm": 24,
    "word_line_pitch_nm": 48, "word_line_width_nm": 24,
    "bit_line_pitch_nm": 72, "bit_line_width_nm": 24, "contact_diameter_nm": 24,
}
DRAM_LOOSE = {
    "kind": "dram", "feature_size_nm": 48,
    "word_line_pitch_nm": 96, "word_line_width_nm": 48,
    "bit_line_pitch_nm": 144, "bit_line_width_nm": 48, "contact_diameter_nm": 48,
}
DRAM_WIDE = {
    "kind": "dram", "feature_size_nm": 60,
    "word_line_pitch_nm": 120, "word_line_width_nm": 56,
    "bit_line_pitch_nm": 180, "bit_line_width_nm": 60, "contact_diameter_nm": 58,
}
DRAM_COMPACT = {
    "kind": "dram", "feature_size_nm": 36,
    "word_line_pitch_nm": 72, "word_line_width_nm": 30,
    "bit_line_pitch_nm": 108, "bit_line_width_nm": 34, "contact_diameter_nm": 30,
}
DRAM_LEGACY = {
    "kind": "dram", "feature_size_nm": 80,
    "word_line_pitch_nm": 160, "word_line_width_nm": 78,
    "bit_line_pitch_nm": 240, "bit_line_width_nm": 80, "contact_diameter_nm": 78,
}
FINFET_10NM = {
    "kind": "finfet", "fin_pitch_nm": 48, "fin_width_nm": 16,
    "gate_pitch_nm": 90, "gate_length_nm": 28, "contact_size_nm": 28,
}
FINFET_7NM = {
    "kind": "finfet", "fin_pitch_nm": 40, "fin_width_nm": 14,
    "gate_pitch_nm": 76, "gate_length_nm": 24, "contact_size_nm": 24,
}
FINFET_14NM = {
    "kind": "finfet", "fin_pitch_nm": 60, "fin_width_nm": 20,
    "gate_pitch_nm": 110, "gate_length_nm": 34, "contact_size_nm": 34,
}
FINFET_22NM = {
    "kind": "finfet", "fin_pitch_nm": 80, "fin_width_nm": 26,
    "gate_pitch_nm": 150, "gate_length_nm": 46, "contact_size_nm": 44,
}
FINFET_28NM = {
    "kind": "finfet", "fin_pitch_nm": 96, "fin_width_nm": 32,
    "gate_pitch_nm": 180, "gate_length_nm": 56, "contact_size_nm": 52,
}
FINFET_45NM = {
    "kind": "finfet", "fin_pitch_nm": 140, "fin_width_nm": 46,
    "gate_pitch_nm": 260, "gate_length_nm": 80, "contact_size_nm": 76,
}

PRESETS = {
    "dram_1x": DRAM_1X, "dram_dense": DRAM_DENSE, "dram_loose": DRAM_LOOSE,
    "dram_wide": DRAM_WIDE, "dram_compact": DRAM_COMPACT, "dram_legacy": DRAM_LEGACY,
    "finfet_10nm": FINFET_10NM, "finfet_7nm": FINFET_7NM, "finfet_14nm": FINFET_14NM,
    "finfet_22nm": FINFET_22NM, "finfet_28nm": FINFET_28NM, "finfet_45nm": FINFET_45NM,
}
DRAM_PRESET_NAMES = [
    "dram_1x", "dram_dense", "dram_loose", "dram_wide", "dram_compact", "dram_legacy",
]
FINFET_PRESET_NAMES = [
    "finfet_10nm", "finfet_7nm", "finfet_14nm", "finfet_22nm", "finfet_28nm", "finfet_45nm",
]
ALL_PRESET_NAMES = DRAM_PRESET_NAMES + FINFET_PRESET_NAMES


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return dict(PRESETS[name])


def presets_for_kind(kind: str):
    names = DRAM_PRESET_NAMES if kind == "dram" else FINFET_PRESET_NAMES
    return [get_preset(n) for n in names]


# =============================================================================
# Layout drawing (1 nm/px)
# =============================================================================
BACKGROUND = 40
WORD_LINE_VAL = 150
BIT_LINE_VAL = 170
CONTACT_VAL = 225
FIN_VAL = 150
GATE_VAL = 170
WIDTH_JITTER_FRACTION = 0.10


def maybe_collapse_gap(gap_nm, threshold_nm, rng, collapse_prob=0.7) -> bool:
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)


def _line_positions(size_px, pitch_nm, rng, jitter_nm) -> np.ndarray:
    positions, pos = [], rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, jitter_nm)
    return np.array(positions)


def _line_mask(size_px, positions, width_nm, collapse_threshold_nm, rng,
               linewidth_bias_nm=0.0, linewidth_scale=1.0):
    """linewidth_scale is the Set B 'polygon scaling' knob: a per-pair
    multiplicative distortion of every drawn feature's width, +/-20% in the
    degraded set. linewidth_bias_nm (additive) is kept for backward compat."""
    mask = np.zeros(size_px, dtype=bool)
    biased = max((width_nm + linewidth_bias_nm) * linewidth_scale, 1.0)
    widths = biased * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION, size=len(positions)))
    widths = np.clip(widths, biased * 0.5, biased * 1.5)
    for i, center in enumerate(positions):
        half = widths[i] / 2.0
        lo, hi = int(round(center - half)), int(round(center + half))
        mask[max(lo, 0):min(hi, size_px)] = True
        if i + 1 < len(positions):
            nh = widths[i + 1] / 2.0
            gap = (positions[i + 1] - nh) - (center + half)
            if maybe_collapse_gap(gap, collapse_threshold_nm, rng):
                blo = int(round(center + half))
                bhi = int(round(positions[i + 1] - nh))
                mask[max(blo, 0):min(bhi, size_px)] = True
    return mask


def _round_corners(canvas, px):
    if px < 0.5:
        return canvas
    k = max(1, int(round(px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)


def generate_dram_canvas(size_px, preset, collapse_threshold_nm, rng,
                         linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                         linewidth_scale=1.0):
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    wl = _line_positions(size_px, preset["word_line_pitch_nm"], rng, 1.5)
    bl = _line_positions(size_px, preset["bit_line_pitch_nm"], rng, 1.5)
    row = _line_mask(size_px, wl, preset["word_line_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    col = _line_mask(size_px, bl, preset["bit_line_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    canvas[row, :] = np.maximum(canvas[row, :], WORD_LINE_VAL)
    canvas[:, col] = np.maximum(canvas[:, col], BIT_LINE_VAL)
    radius0 = max((preset["contact_diameter_nm"] + linewidth_bias_nm) * linewidth_scale, 1.0) / 2.0
    for i, y in enumerate(wl):
        for j, x in enumerate(bl):
            if (i + j) % 2 == 0:
                r = max(1, int(round(radius0 * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)))))
                cv2.circle(canvas, (int(round(x)), int(round(y))), r, CONTACT_VAL, -1)
    return _round_corners(canvas, corner_rounding_px)


def generate_finfet_canvas(size_px, preset, collapse_threshold_nm, rng,
                           linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                           linewidth_scale=1.0):
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    fins = _line_positions(size_px, preset["fin_pitch_nm"], rng, 1.0)
    gates = _line_positions(size_px, preset["gate_pitch_nm"], rng, 1.0)
    col = _line_mask(size_px, fins, preset["fin_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    row = _line_mask(size_px, gates, preset["gate_length_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    canvas[:, col] = np.maximum(canvas[:, col], FIN_VAL)
    canvas[row, :] = np.maximum(canvas[row, :], GATE_VAL)
    half = max(1, int(round(max((preset["contact_size_nm"] + linewidth_bias_nm) * linewidth_scale, 1.0) / 2.0)))
    for i, fx in enumerate(fins):
        for j in range(len(gates) - 1):
            if (i + j) % 2 == 0:
                mid = (gates[j] + gates[j + 1]) / 2.0
                x, y = int(round(fx)), int(round(mid))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, CONTACT_VAL, -1)
    return _round_corners(canvas, corner_rounding_px)


_GENERATORS = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}

STRIP_BASE_VAL = 95
STRIP_LINE_VAL = 128
STRIP_LINE_PITCH_NM = 220
STRIP_LINE_WIDTH_NM = 9


def _strip_routing_texture(size_px, rng):
    canvas = np.full((size_px, size_px), STRIP_BASE_VAL, dtype=np.uint8)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_VAL
            else:
                canvas[:, lo:hi] = STRIP_LINE_VAL
    return canvas


def _zone_grid(size_px, mat_size_nm, strip_width_nm):
    spans, pos, is_mat = [], 0.0, True
    while pos < size_px:
        end = min(pos + (mat_size_nm if is_mat else strip_width_nm), size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos, is_mat = end, not is_mat
    return spans


def generate_zone_canvas(size_px, kind, collapse_threshold_nm, rng,
                         mat_size_nm=2600.0, strip_width_nm=320.0,
                         linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                         linewidth_scale=1.0, preset_names=None):
    generator = _GENERATORS[kind]
    presets = presets_for_kind(kind) if preset_names is None else [get_preset(n) for n in preset_names]
    canvas = _strip_routing_texture(size_px, rng)
    mat_rects, strip_rects = [], []
    for row_is_mat, y0, y1 in _zone_grid(size_px, mat_size_nm, strip_width_nm):
        for col_is_mat, x0, x1 in _zone_grid(size_px, mat_size_nm, strip_width_nm):
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                h, w = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))]
                child = np.random.default_rng(rng.integers(0, 2**31 - 1))
                side = max(h, w)
                mat = generator(side, preset, collapse_threshold_nm, child,
                                linewidth_bias_nm=linewidth_bias_nm,
                                corner_rounding_px=corner_rounding_px,
                                linewidth_scale=linewidth_scale)
                canvas[y0:y1, x0:x1] = mat[:h, :w]
                mat_rects.append((x0, y0, w, h))
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))
    return {"canvas": canvas, "mat_rects": mat_rects, "strip_rects": strip_rects}


# =============================================================================
# Optical RGB layout (Set D bonus only)
# BGR colors so word / bit / contact are different channels, not gray copies.
# =============================================================================
OPT_SUBSTRATE = (55, 45, 35)
OPT_WORD_LINE = (40, 90, 180)
OPT_BIT_LINE = (160, 140, 50)
OPT_CONTACT = (240, 230, 220)
OPT_FIN = (50, 160, 90)
OPT_GATE = (180, 80, 40)
OPT_STRIP_BASE = (90, 85, 70)
OPT_STRIP_LINE = (120, 115, 100)


def _fill_rgb(size_px, bgr):
    canvas = np.empty((size_px, size_px, 3), dtype=np.uint8)
    canvas[:] = bgr
    return canvas


def generate_dram_canvas_rgb(size_px, preset, collapse_threshold_nm, rng,
                             linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                             linewidth_scale=1.0):
    canvas = _fill_rgb(size_px, OPT_SUBSTRATE)
    wl = _line_positions(size_px, preset["word_line_pitch_nm"], rng, 1.5)
    bl = _line_positions(size_px, preset["bit_line_pitch_nm"], rng, 1.5)
    row = _line_mask(size_px, wl, preset["word_line_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    col = _line_mask(size_px, bl, preset["bit_line_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    canvas[row, :] = OPT_WORD_LINE
    canvas[:, col] = OPT_BIT_LINE
    radius0 = max((preset["contact_diameter_nm"] + linewidth_bias_nm) * linewidth_scale, 1.0) / 2.0
    for i, y in enumerate(wl):
        for j, x in enumerate(bl):
            if (i + j) % 2 == 0:
                r = max(1, int(round(radius0 * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)))))
                cv2.circle(canvas, (int(round(x)), int(round(y))), r, OPT_CONTACT, -1)
    return _round_corners(canvas, corner_rounding_px)


def generate_finfet_canvas_rgb(size_px, preset, collapse_threshold_nm, rng,
                               linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                               linewidth_scale=1.0):
    canvas = _fill_rgb(size_px, OPT_SUBSTRATE)
    fins = _line_positions(size_px, preset["fin_pitch_nm"], rng, 1.0)
    gates = _line_positions(size_px, preset["gate_pitch_nm"], rng, 1.0)
    col = _line_mask(size_px, fins, preset["fin_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    row = _line_mask(size_px, gates, preset["gate_length_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm, linewidth_scale)
    canvas[:, col] = OPT_FIN
    canvas[row, :] = OPT_GATE
    half = max(1, int(round(max((preset["contact_size_nm"] + linewidth_bias_nm) * linewidth_scale, 1.0) / 2.0)))
    for i, fx in enumerate(fins):
        for j in range(len(gates) - 1):
            if (i + j) % 2 == 0:
                mid = (gates[j] + gates[j + 1]) / 2.0
                x, y = int(round(fx)), int(round(mid))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, OPT_CONTACT, -1)
    return _round_corners(canvas, corner_rounding_px)


def _strip_routing_texture_rgb(size_px, rng):
    canvas = _fill_rgb(size_px, OPT_STRIP_BASE)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = OPT_STRIP_LINE
            else:
                canvas[:, lo:hi] = OPT_STRIP_LINE
    return canvas


_GENERATORS_RGB = {"dram": generate_dram_canvas_rgb, "finfet": generate_finfet_canvas_rgb}


def generate_zone_canvas_rgb(size_px, kind, collapse_threshold_nm, rng,
                             mat_size_nm=2600.0, strip_width_nm=320.0,
                             linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                             linewidth_scale=1.0, preset_names=None):
    generator = _GENERATORS_RGB[kind]
    presets = presets_for_kind(kind) if preset_names is None else [get_preset(n) for n in preset_names]
    canvas = _strip_routing_texture_rgb(size_px, rng)
    mat_rects, strip_rects = [], []
    for row_is_mat, y0, y1 in _zone_grid(size_px, mat_size_nm, strip_width_nm):
        for col_is_mat, x0, x1 in _zone_grid(size_px, mat_size_nm, strip_width_nm):
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                h, w = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))]
                child = np.random.default_rng(rng.integers(0, 2**31 - 1))
                side = max(h, w)
                mat = generator(side, preset, collapse_threshold_nm, child,
                                linewidth_bias_nm=linewidth_bias_nm,
                                corner_rounding_px=corner_rounding_px,
                                linewidth_scale=linewidth_scale)
                canvas[y0:y1, x0:x1] = mat[:h, :w]
                mat_rects.append((x0, y0, w, h))
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))
    return {"canvas": canvas, "mat_rects": mat_rects, "strip_rects": strip_rects}


def add_optical_channel_noise(img, rng, sigma_bgr=(6.0, 5.0, 7.0)):
    """Independent CMOS noise on B, G, R. Gray SEM path never calls this."""
    out = img.astype(np.float64)
    noise = np.empty_like(out)
    for c, sig in enumerate(sigma_bgr):
        noise[..., c] = rng.normal(0, sig, size=out.shape[:2])
    return np.clip(out + noise, 0, 255).astype(np.uint8)


# =============================================================================
# SEM imaging artifacts
# =============================================================================
def gaussian_psf_blur(img, spot_size_nm, pixel_size_nm, astigmatism_ratio=1.0):
    sx = max(spot_size_nm / pixel_size_nm, 1e-6)
    sy = max(sx * astigmatism_ratio, 1e-6)
    k = max(int(2 * round(3 * max(sx, sy)) + 1), 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sx, sigmaY=sy)


def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    factor = 1.0 - strength * (r ** 2)
    if img.ndim == 3:
        factor = factor[..., None]
    return np.clip(img.astype(np.float64) * factor, 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img
    return np.clip(np.power(np.clip(img.astype(np.float64) / 255.0, 0, 1), gamma) * 255.0,
                   0, 255).astype(np.uint8)


def apply_barrel_distortion(img, k):
    """'Scan distortion' in the Phase 2 Set B description."""
    if k == 0.0:
        return img
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    factor = 1.0 + k * (nx ** 2 + ny ** 2)
    return cv2.remap(img, (nx * factor) * cx + cx, (ny * factor) * cy + cy,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def forward_barrel_point(x, y, k, h, w):
    """Where a feature at source (x, y) ends up after apply_barrel_distortion().

    That remap is a backward map: the destination at normalised radius d reads
    the source at radius s = d * (1 + k*d^2). Pushing a label forward therefore
    means solving d + k*d^3 = s for d, which is monotonic in d for k >= 0, so a
    few Newton steps from d = s converge. Direction is preserved because both
    axes are scaled by the same factor."""
    if k == 0.0:
        return float(x), float(y)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    nx, ny = (float(x) - cx) / cx, (float(y) - cy) / cy
    s = math.hypot(nx, ny)
    if s < 1e-12:
        return float(x), float(y)
    d = s
    for _ in range(40):
        f = d + k * d ** 3 - s
        if abs(f) < 1e-12:
            break
        d -= f / (1.0 + 3.0 * k * d * d)
    ratio = d / s
    return (nx * ratio) * cx + cx, (ny * ratio) * cy + cy


@dataclass
class SearchGeometry:
    """The label-moving warps image_search() applied, in pipeline order, so a
    pose can be pushed through the identical maps (spec R5)."""
    drift_shift: "np.ndarray | None" = None
    barrel_k: float = 0.0
    size_px: int = 1000

    def forward(self, x, y):
        if self.drift_shift is not None:
            x, y = forward_drift_point(x, y, self.drift_shift)
        return forward_barrel_point(x, y, self.barrel_k, self.size_px, self.size_px)


def add_charging_streaks(img, streak_prob, intensity, rng):
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape[:2]
    out = img.astype(np.float64)
    n = rng.poisson(max(streak_prob * (h / 100.0), 0))
    for _ in range(n):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.uint8)


def downsample_to_size(img, out_size):
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)


def rotate_about_center(img, angle_deg, gx, gy):
    """cv2's angle convention is CCW-positive, matching the addendum's
    theta definition (rotation about the match centre, CCW positive)."""
    if angle_deg == 0.0:
        return img, float(gx), float(gy)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    nx = float(M[0, 0] * gx + M[0, 1] * gy + M[0, 2])
    ny = float(M[1, 0] * gx + M[1, 1] * gy + M[1, 2])
    return warped, nx, ny


def center_rotation_matrix(size_px, angle_deg):
    """The matrix rotate_about_center() would use, so an image can be warped
    once and any number of labelled points pushed through the same map."""
    return cv2.getRotationMatrix2D((size_px / 2.0, size_px / 2.0), angle_deg, 1.0)


def apply_affine_point(M, gx, gy):
    return (float(M[0, 0] * gx + M[0, 1] * gy + M[0, 2]),
            float(M[1, 0] * gx + M[1, 1] * gy + M[1, 2]))


class LabelVerificationError(RuntimeError):
    """Raised when no crop on a canvas produces a reproducible label."""


def _png_roundtrip(img):
    """Quantise exactly as cv2.imwrite/imread would. PNG is lossless for uint8,
    so this is byte-identical to the file that ships -- and the gate has to see
    the quantised image, because rounding alone can move a correlation peak to
    a neighbouring lattice site on a periodic array."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise LabelVerificationError("PNG encode failed during verification")
    flag = cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(buf, flag)


def render_probe_template(reference_img, scale_ratio, rotation_deg):
    """Build the template a *correct solver* would build: plain area-resize of
    the shipped reference onto the search grid, then a rotation warp.

    Deliberately independent of image_search()'s blur-then-downsample path, so
    the gate cannot pass merely by sharing the generator's resampler."""
    if reference_img.ndim == 3:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    tw = max(int(round(REFERENCE_SIZE_PX / float(scale_ratio))), 8)
    tmpl = cv2.resize(reference_img, (tw, tw), interpolation=cv2.INTER_AREA)
    if rotation_deg != 0.0:
        M = cv2.getRotationMatrix2D((tw / 2.0 - 0.5, tw / 2.0 - 0.5), rotation_deg, 1.0)
        tmpl = cv2.warpAffine(tmpl, M, (tw, tw), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return tmpl


def verify_label(reference_img, search_img, gt_x, gt_y, scale_ratio, rotation_deg):
    """Template-match the shipped reference against the shipped search image at
    the labelled pose.

    Returns (err_px, margin):
      err_px  distance from the *global* correlation peak to the label. A large
              value paired with a high peak is a broken label, not a hard pair.
      margin  peak score minus the best competing peak outside a local
              exclusion window -- how far the true site leads its aliases.
    """
    if search_img.ndim == 3:
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    tmpl = render_probe_template(reference_img, scale_ratio, rotation_deg)
    tw = tmpl.shape[0]
    if tw > search_img.shape[0] or tw > search_img.shape[1]:
        return float("inf"), 0.0
    resp = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (xi, yi) = cv2.minMaxLoc(resp)
    err = float(np.hypot(xi + tw / 2.0 - gt_x, yi + tw / 2.0 - gt_y))

    excl = max(int(round(tw * 0.5)), 8)
    masked = resp.copy()
    masked[max(yi - excl, 0):yi + excl + 1, max(xi - excl, 0):xi + excl + 1] = -1.0
    runner_up = float(masked.max())
    return err, float(peak - runner_up)


def sample_drift_shift(h, shear_amplitude_px, jitter_std_px, rng):
    """Per-row horizontal shift: a linear shear plus white scan jitter.

    Returned so callers can push a label through the same warp -- see
    forward_drift_point(). Row order matches the image, and the shift is a
    function of the row alone, so the map has no y coupling."""
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0 else np.zeros(h)
    return (shear + jitter).astype(np.float32)


def apply_drift_shift(img, shift):
    h, w = img.shape[:2]
    map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def forward_drift_point(x, y, shift):
    """Where a feature at source (x, y) ends up after apply_drift_shift().

    cv2.remap is a backward map: dst(x, y) = src(x + shift[y], y). So source
    column x on row y is read at destination column x - shift[y]. The shift is
    interpolated between rows, since a label centre need not be on one."""
    h = len(shift)
    yc = float(np.clip(y, 0, h - 1))
    lo = int(np.floor(yc))
    hi = min(lo + 1, h - 1)
    s = float(shift[lo] + (shift[hi] - shift[lo]) * (yc - lo))
    return float(x) - s, float(y)


def apply_raster_drift(img, shear_amplitude_px, jitter_std_px, rng):
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    return apply_drift_shift(img, sample_drift_shift(img.shape[0], shear_amplitude_px,
                                                     jitter_std_px, rng))


def add_shot_noise(img, dose, rng):
    img_f = img.astype(np.float64)
    noisy = rng.poisson(np.clip(img_f / 255.0 * dose, 0, None)).astype(np.float64)
    return np.clip(noisy / dose * 255.0, 0, 255).astype(np.uint8)


def add_detector_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, size=img.shape),
                   0, 255).astype(np.uint8)


def add_speckle_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) * (1.0 + rng.normal(0, sigma, size=img.shape)),
                   0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img, prob, rng):
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def image_reference(crop, pixel_size_nm, spot_size_nm, dose, rng,
                    detector_noise_sigma=2.0, drift_jitter_px=0.2,
                    astigmatism_ratio=1.0, vignette_strength=0.0, gamma=1.0,
                    barrel_distortion_k=0.0, charging_streak_prob=0.0,
                    charging_streak_intensity=0.0, speckle_sigma=0.0,
                    salt_pepper_prob=0.0):
    img = gaussian_psf_blur(crop, spot_size_nm, pixel_size_nm, astigmatism_ratio)
    img = apply_raster_drift(img, 0.0, drift_jitter_px, rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img


def image_search(full_canvas, pixel_size_ref_nm, pixel_size_search_nm, spot_size_nm,
                 dose, rng, shear_amplitude_px=1.5, drift_jitter_px=0.5,
                 detector_noise_sigma=5.0, astigmatism_ratio=1.0, vignette_strength=0.0,
                 gamma=1.0, barrel_distortion_k=0.0, charging_streak_prob=0.0,
                 charging_streak_intensity=0.0, speckle_sigma=0.0, salt_pepper_prob=0.0,
                 search_size_px=1000, return_geometry=False):
    blurred = gaussian_psf_blur(full_canvas, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio)
    down = blurred if blurred.shape[0] == search_size_px else downsample_to_size(blurred, search_size_px)
    # Drift and barrel move every pixel, so the exact maps are handed back for
    # the label to travel through as well.
    shift = None
    if shear_amplitude_px != 0 or drift_jitter_px != 0:
        shift = sample_drift_shift(down.shape[0], shear_amplitude_px, drift_jitter_px, rng)
        img = apply_drift_shift(down, shift)
    else:
        img = down
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    if return_geometry:
        return img, SearchGeometry(drift_shift=shift, barrel_k=barrel_distortion_k,
                                   size_px=img.shape[0])
    return img


# =============================================================================
# One sample
# =============================================================================
@dataclass
class GenerationParams:
    beam_spot_size_nm: float = 5.0
    collapse_threshold_nm: float = 10.0
    dose_reference: float = 2000.0
    dose_search: float = 200.0
    shear_amplitude_px: float = 1.5
    drift_jitter_px: float = 0.5
    detector_noise_sigma_ref: float = 2.0
    detector_noise_sigma_search: float = 5.0
    astigmatism_ratio: float = 1.0
    vignette_strength: float = 0.0
    gamma: float = 1.0
    barrel_distortion_k: float = 0.0     # "scan distortion" knob, Set B
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    speckle_sigma: float = 0.0
    salt_pepper_prob: float = 0.0
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    boundary_bias: float = 0.35
    linewidth_bias_nm: float = 0.0
    linewidth_scale: float = 1.0         # "polygon scaling", Set B, +/-20%
    corner_rounding_px: float = 0.0
    scale_ratio: float = 10.0
    rotation_deg: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def fine_canvas_size_px(scale_ratio: float) -> int:
    return max(int(round(REFERENCE_SIZE_PX * float(scale_ratio))), REFERENCE_SIZE_PX)


def _pick_crop_origin(zone_result, params, rng, canvas_size):
    max_offset = canvas_size - REFERENCE_SIZE_PX
    strips = zone_result.get("strip_rects") or []
    if strips and rng.random() < params.boundary_bias:
        sx, sy, sw, sh = strips[int(rng.integers(0, len(strips)))]
        x0 = int(np.clip(sx + sw / 2.0 - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250),
                         0, max_offset))
        y0 = int(np.clip(sy + sh / 2.0 - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250),
                         0, max_offset))
        return x0, y0
    return int(rng.integers(0, max_offset + 1)), int(rng.integers(0, max_offset + 1))


def generate_sample(architecture, rng, params: GenerationParams,
                    verify: bool = False) -> dict:
    """Present-instance sample (Set A / Set B). found=1.

    The search image depends only on the canvas, not on where the reference was
    cropped from it, so it is rendered once and the crop is resampled until the
    label clears the verification gate. Retries are cheap for that reason --
    canvas synthesis, the expensive part, is not repeated."""
    preset = get_preset(architecture)
    zone = generate_zone_canvas(
        fine_canvas_size_px(params.scale_ratio), preset["kind"],
        params.collapse_threshold_nm, rng,
        mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
        linewidth_scale=params.linewidth_scale,
    )
    canvas = zone["canvas"]
    n = canvas.shape[0]
    search_img, geom = image_search(
        canvas, PIXEL_SIZE_REF_NM, n / float(SEARCH_SIZE_PX), params.beam_spot_size_nm,
        params.dose_search, rng, return_geometry=True,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        search_size_px=SEARCH_SIZE_PX,
    )
    scale = n / float(SEARCH_SIZE_PX)
    box = REFERENCE_SIZE_PX / scale
    M = center_rotation_matrix(SEARCH_SIZE_PX, params.rotation_deg)
    if params.rotation_deg != 0.0:
        search_img = cv2.warpAffine(search_img, M, (SEARCH_SIZE_PX, SEARCH_SIZE_PX),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    realized = replace(params, scale_ratio=scale)
    search_shipped = _png_roundtrip(search_img) if verify else None

    best = None
    for attempt in range(1, (VERIFY_CROP_ATTEMPTS if verify else 1) + 1):
        x0, y0 = _pick_crop_origin(zone, params, rng, n)
        crop = canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]
        reference_img = image_reference(
            crop, PIXEL_SIZE_REF_NM, params.beam_spot_size_nm, params.dose_reference, rng,
            detector_noise_sigma=params.detector_noise_sigma_ref,
            drift_jitter_px=params.drift_jitter_px * 0.2,
            astigmatism_ratio=params.astigmatism_ratio,
            vignette_strength=params.vignette_strength * 0.5,
            gamma=params.gamma,
            barrel_distortion_k=params.barrel_distortion_k * 0.3,
            charging_streak_prob=params.charging_streak_prob,
            charging_streak_intensity=params.charging_streak_intensity,
            speckle_sigma=params.speckle_sigma,
            salt_pepper_prob=params.salt_pepper_prob,
        )
        # Pose -> drift -> barrel -> rotation, the same order image_search()
        # and the rotation above applied them in.
        cx, cy = geom.forward(x0 / scale + box / 2.0, y0 / scale + box / 2.0)
        cx, cy = apply_affine_point(M, cx, cy)
        cand = {
            "reference_img": reference_img, "search_img": search_img,
            "found": 1, "gt_x": cx, "gt_y": cy, "gt_theta": params.rotation_deg,
            "gt_scale": scale, "architecture": architecture,
            "params": realized.as_dict(), "canvas_size_px": n,
            "crop_attempts": attempt, "verify_err_px": None, "verify_margin": None,
        }
        if not verify:
            return cand

        err, margin = verify_label(_png_roundtrip(reference_img), search_shipped,
                                   cx, cy, scale, params.rotation_deg)
        cand["verify_err_px"], cand["verify_margin"] = err, margin
        if err > VERIFY_MAX_ERR_PX:
            continue
        if margin >= VERIFY_MARGIN_PREFER:
            return cand
        # Heavy degradation compresses the margin, so keep the strongest
        # near-miss and fall back to the floor rather than discarding the pair.
        if best is None or margin > best["verify_margin"]:
            best = cand

    if best is not None and best["verify_margin"] >= VERIFY_MARGIN_FLOOR:
        return best
    raise LabelVerificationError(
        f"{architecture}: no crop in {VERIFY_CROP_ATTEMPTS} attempts reproduced its "
        f"label (best margin={best['verify_margin']:.4f} err={best['verify_err_px']:.2f}px)"
        if best else
        f"{architecture}: no crop in {VERIFY_CROP_ATTEMPTS} attempts landed within "
        f"{VERIFY_MAX_ERR_PX} px of its label")


def generate_sample_absent(architecture, rng, params: GenerationParams,
                           verify: bool = False) -> dict:
    """Set C: no true instance. The reference is a real crop from canvas A;
    the search image comes from an *independently generated* canvas B of the
    same architecture family (same kind, same-ish pitches) so it is
    'plausible and periodically similar' but contains no true match.
    found=0, and per the output contract, pose columns are written as 0.
    """
    preset = get_preset(architecture)
    kind = preset["kind"]

    # Reference: an ordinary crop, exactly as in the present-instance path.
    zone_ref = generate_zone_canvas(
        fine_canvas_size_px(params.scale_ratio), kind,
        params.collapse_threshold_nm, rng,
        mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
        linewidth_scale=params.linewidth_scale,
    )
    canvas_ref = zone_ref["canvas"]
    n_ref = canvas_ref.shape[0]
    x0, y0 = _pick_crop_origin(zone_ref, params, rng, n_ref)
    crop = canvas_ref[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]
    reference_img = image_reference(
        crop, PIXEL_SIZE_REF_NM, params.beam_spot_size_nm, params.dose_reference, rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        gamma=params.gamma,
        # The reference is a high-magnification view of a small area, so scan
        # distortion across it is negligible. Giving it a fraction of the
        # search's barrel -- about its own, different centre -- only deformed
        # the template relative to the search content, even with a exact label.
        barrel_distortion_k=0.0,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
    )

    # Search: a *different* die region -- independently generated canvas,
    # same architecture kind (periodically similar), unrelated random layout,
    # so nothing in it truly matches the reference crop above.
    #
    # IMPORTANT: mix across the full preset family here, exactly like the
    # present-pair canvases do (generate_sample leaves preset_names=None).
    # Forcing a single preset for the whole decoy canvas would make Set C
    # texturally more uniform than Set A/B -- a classifier could then learn
    # "uniform texture => absent" as a shortcut instead of actually
    # comparing to the reference, which would inflate rejection F1 here
    # without generalizing to a real blind Set C.
    zone_decoy = generate_zone_canvas(
        fine_canvas_size_px(params.scale_ratio), kind,
        params.collapse_threshold_nm, rng,
        mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
        linewidth_scale=params.linewidth_scale,
        preset_names=None,
    )
    canvas_decoy = zone_decoy["canvas"]
    n_decoy = canvas_decoy.shape[0]

    search_img = image_search(
        canvas_decoy, PIXEL_SIZE_REF_NM, n_decoy / float(SEARCH_SIZE_PX), params.beam_spot_size_nm,
        params.dose_search, rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        search_size_px=SEARCH_SIZE_PX,
    )
    # Rotation is still physically applied to the search image (the tool
    # doesn't know in advance there's no match) but there is no meaningful
    # ground-truth location, so per the output contract we report 0s.
    search_img, _, _ = rotate_about_center(search_img, params.rotation_deg, 0.0, 0.0)
    realized = replace(params, scale_ratio=n_decoy / float(SEARCH_SIZE_PX))
    return {
        "reference_img": reference_img, "search_img": search_img,
        "found": 0, "gt_x": 0.0, "gt_y": 0.0, "gt_theta": 0.0, "gt_scale": 0.0,
        "architecture": architecture, "params": realized.as_dict(),
    }


def generate_sample_optical(architecture, rng, params: GenerationParams,
                            verify: bool = False) -> dict:
    """Set D bonus: same geometry/GT logic as generate_sample, true BGR (HxWx3)."""
    preset = get_preset(architecture)
    opt = replace(params, charging_streak_prob=0.0, charging_streak_intensity=0.0)
    zone = generate_zone_canvas_rgb(
        fine_canvas_size_px(opt.scale_ratio), preset["kind"],
        opt.collapse_threshold_nm, rng,
        mat_size_nm=opt.mat_size_nm, strip_width_nm=opt.strip_width_nm,
        linewidth_bias_nm=opt.linewidth_bias_nm,
        corner_rounding_px=opt.corner_rounding_px,
        linewidth_scale=opt.linewidth_scale,
    )
    canvas = zone["canvas"]
    n = canvas.shape[0]
    x0, y0 = _pick_crop_origin(zone, opt, rng, n)
    crop = canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    reference_img = image_reference(
        crop, PIXEL_SIZE_REF_NM, opt.beam_spot_size_nm, opt.dose_reference, rng,
        detector_noise_sigma=opt.detector_noise_sigma_ref,
        drift_jitter_px=opt.drift_jitter_px * 0.2,
        astigmatism_ratio=opt.astigmatism_ratio,
        vignette_strength=max(opt.vignette_strength, 0.12),
        gamma=opt.gamma,
        barrel_distortion_k=0.0,
        charging_streak_prob=0.0,
        charging_streak_intensity=0.0,
        speckle_sigma=opt.speckle_sigma,
        salt_pepper_prob=opt.salt_pepper_prob,
    )
    search_img, geom = image_search(
        canvas, PIXEL_SIZE_REF_NM, n / float(SEARCH_SIZE_PX), opt.beam_spot_size_nm,
        opt.dose_search, rng, return_geometry=True,
        shear_amplitude_px=opt.shear_amplitude_px,
        drift_jitter_px=opt.drift_jitter_px,
        detector_noise_sigma=opt.detector_noise_sigma_search,
        astigmatism_ratio=opt.astigmatism_ratio,
        vignette_strength=max(opt.vignette_strength, 0.18),
        gamma=opt.gamma,
        barrel_distortion_k=opt.barrel_distortion_k,
        charging_streak_prob=0.0,
        charging_streak_intensity=0.0,
        speckle_sigma=opt.speckle_sigma,
        salt_pepper_prob=opt.salt_pepper_prob,
        search_size_px=SEARCH_SIZE_PX,
    )
    reference_img = add_optical_channel_noise(reference_img, rng, (3.0, 2.5, 3.5))
    search_img = add_optical_channel_noise(search_img, rng, (6.0, 5.0, 7.0))
    scale = n / float(SEARCH_SIZE_PX)
    box = REFERENCE_SIZE_PX / scale
    cx, cy = geom.forward(x0 / scale + box / 2.0, y0 / scale + box / 2.0)
    search_img, cx, cy = rotate_about_center(search_img, opt.rotation_deg, cx, cy)
    realized = replace(opt, scale_ratio=scale)
    return {
        "reference_img": reference_img, "search_img": search_img,
        "found": 1, "gt_x": cx, "gt_y": cy, "gt_theta": opt.rotation_deg,
        "gt_scale": scale, "architecture": architecture, "params": realized.as_dict(),
    }


# =============================================================================
# Per-pair parameter sampling
# =============================================================================
def _base_geometry(rng, base: GenerationParams) -> GenerationParams:
    """Scale/rotation shared by Sets A, B, D: full Phase 2 disclosed range."""
    p = replace(base)
    p.scale_ratio = float(rng.uniform(*SCALE_RATIO_RANGE))
    p.rotation_deg = float(rng.uniform(*ROTATION_DEG_RANGE))
    p.speckle_sigma = 0.0
    p.dose_reference = float(rng.uniform(1500.0, 3000.0))
    p.detector_noise_sigma_ref = float(rng.uniform(1.0, 3.0))
    p.astigmatism_ratio = float(rng.uniform(0.88, 1.12))
    return p


def sample_params_set_a(rng, base: GenerationParams) -> GenerationParams:
    """Nominal pose: noise comparable to the Phase 1 sample prompt.
    Full [8,12]x / +/-5deg range, but no scan distortion / defocus /
    polygon scaling -- those are reserved for Set B."""
    p = _base_geometry(rng, base)
    p.dose_search = float(rng.uniform(150.0, 350.0))
    p.detector_noise_sigma_search = float(rng.uniform(4.0, 6.5))
    p.drift_jitter_px = float(rng.uniform(0.30, 0.70))
    p.shear_amplitude_px = float(rng.uniform(0.40, 1.20))
    p.beam_spot_size_nm = float(rng.uniform(3.0, 5.5))
    p.gamma = float(rng.uniform(0.95, 1.05))
    p.barrel_distortion_k = 0.0
    p.linewidth_scale = 1.0
    p.charging_streak_prob = 0.0
    p.charging_streak_intensity = 0.0
    p.salt_pepper_prob = 0.0
    p.boundary_bias = 0.45
    return p


def sample_params_set_b(rng, base: GenerationParams, severity=None) -> GenerationParams:
    """Degraded present pairs: four monotonic severity levels, harsher than the
    organizer calibration band.

    Scan distortion and raster drift are back, and the label is pushed through
    both maps in generate_sample() (spec R5). Before that correction existed
    they slid the true instance up to ~70 px away from the labelled centre
    while the label stayed put, which silently poisoned training: the correct
    window fell outside POS_RADIUS and became a labelled hard negative.

    What is *not* fixable by a label is deformation inside the template window,
    since barrel is not a rigid transform. k is therefore kept in the disclosed
    band so intra-window stretch stays small enough for a rigid matcher, and
    the read-back gate measures whatever residual remains."""
    p = _base_geometry(rng, base)
    if severity is None:
        severity = DEGRADATION_LEVELS[int(rng.integers(0, len(DEGRADATION_LEVELS)))]
    severity = float(severity)

    p.dose_search = float(rng.uniform(60.0, 140.0) / severity)
    p.detector_noise_sigma_search = float(rng.uniform(7.0, 12.0) * severity)
    p.speckle_sigma = float(rng.uniform(0.04, 0.14))
    p.salt_pepper_prob = float(rng.uniform(0.0010, 0.0060) * severity)
    p.charging_streak_prob = float(rng.uniform(0.50, 1.60) * severity)
    p.charging_streak_intensity = float(rng.uniform(0.40, 0.90))
    p.gamma = float(rng.uniform(0.80, 1.20))
    p.vignette_strength = float(rng.uniform(0.10, 0.35))
    p.linewidth_scale = float(rng.uniform(0.88, 1.12))
    # Defocus is capped: dram_dense has a 48 nm word-line pitch, so a spot size
    # much past ~11 nm erases the pattern outright instead of testing robustness.
    p.beam_spot_size_nm = float(rng.uniform(4.0, 7.5) * min(severity, 1.5))

    # Geometry: label-corrected, so these are free to be aggressive again.
    # Shear is a smooth ramp, exactly correctable at the centre and leaving
    # only ~0.1*shear px of residual tilt across a template window.
    p.shear_amplitude_px = float(rng.uniform(0.85, 2.20) * severity)
    # Jitter is white per row, so the centre is corrected but the rows still
    # wobble against each other -- genuine, uncorrectable scan noise.
    p.drift_jitter_px = float(rng.uniform(0.35, 0.78) * severity)
    p.barrel_distortion_k = float(rng.uniform(0.003, 0.020) * severity)
    p.astigmatism_ratio = float(rng.uniform(1.0, 1.0 + 0.15 * severity))
    p.boundary_bias = 0.35
    return p


def sample_params_set_c(rng, base: GenerationParams) -> GenerationParams:
    """Absent: reference is real, search comes from a decoy region. Noise on
    the decoy search is drawn from the same range as Set A so the pair isn't
    trivially flaggable by noise level alone."""
    return sample_params_set_a(rng, base)


def sample_params_set_d(rng, base: GenerationParams) -> GenerationParams:
    """Optical bonus: same geometry range, Set-A-like noise (defocus/charging
    read out differently on an optical microscope so kept mild)."""
    p = sample_params_set_a(rng, base)
    p.vignette_strength = float(rng.uniform(0.08, 0.18))
    return p


# =============================================================================
# CLI / dataset assembly
# =============================================================================
def _write_pair(ref_dir, search_dir, idx, sample, optical=False):
    ref_path = os.path.join(ref_dir, f"{idx:05d}.png")
    search_path = os.path.join(search_dir, f"{idx:05d}.png")
    cv2.imwrite(ref_path, sample["reference_img"])
    cv2.imwrite(search_path, sample["search_img"])
    return ref_path, search_path


MANIFEST_FIELDS = [
    "pair_id", "set", "architecture", "channels", "present", "severity",
    "zoom", "theta", "canvas_size_px", "verify_err_px", "verify_margin",
    "crop_attempts", "canvas_attempts", "dose_search",
    "detector_noise_sigma_search", "beam_spot_size_nm", "drift_jitter_px",
    "shear_amplitude_px", "barrel_distortion_k", "charging_streak_prob",
    "salt_pepper_prob", "speckle_sigma", "gamma", "vignette_strength",
    "linewidth_scale", "reference_path", "search_path",
]


def _manifest_row(pair_id, set_name, sample, params, severity, optical,
                  ref_path, search_path, canvas_attempts):
    err, margin = sample.get("verify_err_px"), sample.get("verify_margin")
    return {
        "pair_id": pair_id, "set": set_name,
        "architecture": sample["architecture"], "channels": 3 if optical else 1,
        "present": sample["found"],
        "severity": "" if severity is None else f"{severity:.3f}",
        "zoom": f"{sample['gt_scale']:.5f}", "theta": f"{sample['gt_theta']:.5f}",
        "canvas_size_px": sample.get("canvas_size_px", ""),
        "verify_err_px": "" if err is None else f"{err:.4f}",
        "verify_margin": "" if margin is None else f"{margin:.5f}",
        "crop_attempts": sample.get("crop_attempts", ""),
        "canvas_attempts": canvas_attempts,
        "dose_search": f"{params.dose_search:.3f}",
        "detector_noise_sigma_search": f"{params.detector_noise_sigma_search:.3f}",
        "beam_spot_size_nm": f"{params.beam_spot_size_nm:.3f}",
        "drift_jitter_px": f"{params.drift_jitter_px:.4f}",
        "shear_amplitude_px": f"{params.shear_amplitude_px:.4f}",
        "barrel_distortion_k": f"{params.barrel_distortion_k:.5f}",
        "charging_streak_prob": f"{params.charging_streak_prob:.4f}",
        "salt_pepper_prob": f"{params.salt_pepper_prob:.6f}",
        "speckle_sigma": f"{params.speckle_sigma:.4f}",
        "gamma": f"{params.gamma:.4f}",
        "vignette_strength": f"{params.vignette_strength:.4f}",
        "linewidth_scale": f"{params.linewidth_scale:.4f}",
        "reference_path": ref_path, "search_path": search_path,
    }


def build_set(set_name, n, sampler, maker, architectures, rng, base_params,
             ref_dir, search_dir, start_id, optical=False, severity_levels=None,
             verify=False):
    rows_pairs, rows_truth, rows_manifest = [], [], []
    for i in range(n):
        architecture = architectures[int(rng.integers(0, len(architectures)))]
        severity = None
        if severity_levels is None:
            pair_params = sampler(rng, base_params)
        else:
            severity = float(severity_levels[i % len(severity_levels)])
            pair_params = sampler(rng, base_params, severity)
        pair_id = start_id + i

        # A canvas whose crops are all ambiguous is discarded wholesale; only
        # after VERIFY_CANVAS_ATTEMPTS do we abort, so a broken label can never
        # be written silently.
        sample = ref_path = search_path = None
        last_error = None
        canvas_attempts = 0
        for canvas_attempts in range(1, (VERIFY_CANVAS_ATTEMPTS if verify else 1) + 1):
            try:
                sample = maker(architecture, rng, pair_params, verify=verify)
                ref_path, search_path = _write_pair(ref_dir, search_dir, pair_id,
                                                   sample, optical)
                if verify and sample.get("verify_err_px") is not None:
                    # Gate the artifact that ships, not the in-memory array.
                    read_flag = cv2.IMREAD_COLOR if optical else cv2.IMREAD_GRAYSCALE
                    err, margin = verify_label(
                        cv2.imread(ref_path, read_flag),
                        cv2.imread(search_path, read_flag),
                        sample["gt_x"], sample["gt_y"],
                        sample["gt_scale"], sample["gt_theta"])
                    if err > VERIFY_MAX_ERR_PX or margin < VERIFY_MARGIN_FLOOR:
                        raise LabelVerificationError(
                            f"pair {pair_id} failed on re-read: err={err:.2f}px "
                            f"margin={margin:.4f}")
                    sample["verify_err_px"], sample["verify_margin"] = err, margin
                break
            except LabelVerificationError as exc:
                last_error = exc
                sample = None
        if sample is None:
            raise LabelVerificationError(
                f"[{set_name}] pair {pair_id}: {VERIFY_CANVAS_ATTEMPTS} canvases "
                f"failed verification -- last: {last_error}")

        rows_pairs.append({
            "pair_id": pair_id, "set": set_name,
            "reference_path": ref_path, "search_path": search_path,
        })
        rows_truth.append({
            "pair_id": pair_id, "set": set_name, "found": sample["found"],
            "gt_x": sample["gt_x"], "gt_y": sample["gt_y"],
            "gt_theta": sample["gt_theta"], "gt_scale": sample["gt_scale"],
            "architecture": sample["architecture"],
        })
        rows_manifest.append(_manifest_row(pair_id, set_name, sample, pair_params,
                                           severity, optical, ref_path, search_path,
                                           canvas_attempts))
        gate = ""
        if sample.get("verify_err_px") is not None:
            gate = (f" verify={sample['verify_err_px']:.2f}px "
                    f"margin={sample['verify_margin']:.3f} "
                    f"tries={sample['crop_attempts']}")
        print(f"[{set_name}] {pair_id:04d}  found={sample['found']}  "
              f"arch={sample['architecture']:12s} "
              f"scale={pair_params.scale_ratio:.2f} rot={pair_params.rotation_deg:+.2f}"
              f"{gate}")
    return rows_pairs, rows_truth, rows_manifest, start_id + n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-a", type=int, default=70, help="nominal pose pairs")
    p.add_argument("--set-b", type=int, default=70, help="degraded pairs")
    p.add_argument("--set-c", type=int, default=40, help="absent pairs")
    p.add_argument("--set-d", type=int, default=20, help="optical bonus pairs")
    p.add_argument("--architectures", nargs="+", default=ALL_PRESET_NAMES,
                   choices=list(PRESETS.keys()))
    p.add_argument("--output-dir", default="./dataset_phase2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-verify", action="store_true",
                   help="skip the read-back label gate (debugging only -- "
                        "labels are then unverified)")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    base_params = GenerationParams()

    gray_dir = os.path.join(args.output_dir, "grayscale")
    gray_ref, gray_search = os.path.join(gray_dir, "reference"), os.path.join(gray_dir, "search")
    os.makedirs(gray_ref, exist_ok=True)
    os.makedirs(gray_search, exist_ok=True)
    if args.set_d > 0:
        opt_dir = os.path.join(args.output_dir, "optical")
        opt_ref, opt_search = os.path.join(opt_dir, "reference"), os.path.join(opt_dir, "search")
        os.makedirs(opt_ref, exist_ok=True)
        os.makedirs(opt_search, exist_ok=True)

    print(f"Phase 2 blind dataset -> {args.output_dir}  seed={args.seed}")
    print(f"Set A(nominal)={args.set_a}  Set B(degraded)={args.set_b}  "
          f"Set C(absent)={args.set_c}  Set D(optical bonus)={args.set_d}")

    verify = not args.skip_verify
    print(f"Label gate: {'ON' if verify else 'OFF (labels unverified!)'}  "
          f"peak within {VERIFY_MAX_ERR_PX} px, margin >= {VERIFY_MARGIN_FLOOR} "
          f"(target {VERIFY_MARGIN_PREFER})")

    pairs, truth, manifest = [], [], []
    next_id = 0

    a_pairs, a_truth, a_manifest, next_id = build_set(
        "A", args.set_a, sample_params_set_a, generate_sample,
        args.architectures, rng, base_params, gray_ref, gray_search, next_id,
        verify=verify)
    pairs += a_pairs
    truth += a_truth
    manifest += a_manifest

    b_pairs, b_truth, b_manifest, next_id = build_set(
        "B", args.set_b, sample_params_set_b, generate_sample,
        args.architectures, rng, base_params, gray_ref, gray_search, next_id,
        severity_levels=DEGRADATION_LEVELS, verify=verify)
    pairs += b_pairs
    truth += b_truth
    manifest += b_manifest

    c_pairs, c_truth, c_manifest, next_id = build_set(
        "C", args.set_c, sample_params_set_c, generate_sample_absent,
        args.architectures, rng, base_params, gray_ref, gray_search, next_id)
    pairs += c_pairs
    truth += c_truth
    manifest += c_manifest

    if args.set_d > 0:
        d_pairs, d_truth, d_manifest, next_id = build_set(
            "D", args.set_d, sample_params_set_d, generate_sample_optical,
            args.architectures, rng, base_params, opt_ref, opt_search, next_id, optical=True)
        pairs += d_pairs
        truth += d_truth
        manifest += d_manifest

    with open(os.path.join(args.output_dir, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "set", "reference_path", "search_path"])
        w.writeheader()
        w.writerows(pairs)

    with open(os.path.join(args.output_dir, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "set", "found", "gt_x", "gt_y",
                                          "gt_theta", "gt_scale", "architecture"])
        w.writeheader()
        w.writerows(truth)

    with open(os.path.join(args.output_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)

    n_gray = args.set_a + args.set_b + args.set_c
    print(f"Wrote {len(pairs)} pairs ({n_gray} grayscale for rejection F1, "
          f"{args.set_d} optical bonus) to {args.output_dir}")
    print("pairs.csv        -> what a team would receive (no ground truth)")
    print("ground_truth.csv -> held-out answer key for scoring / threshold tuning")
    print("manifest.csv     -> per-pair generation + label-verification audit")

    errs = [float(r["verify_err_px"]) for r in manifest if r["verify_err_px"] != ""]
    margins = [float(r["verify_margin"]) for r in manifest if r["verify_margin"] != ""]
    if errs:
        tries = [int(r["crop_attempts"]) for r in manifest if r["verify_err_px"] != ""]
        print(f"Label gate: {len(errs)} present pairs verified  "
              f"err mean={np.mean(errs):.3f} max={np.max(errs):.3f} px  "
              f"margin mean={np.mean(margins):.3f} min={np.min(margins):.3f}  "
              f"crop attempts mean={np.mean(tries):.2f} max={np.max(tries)}")


if __name__ == "__main__":
    main()