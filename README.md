# DriftLoc — Semicon Phase 2

Pattern localisation for DRAM die images.  
Given a 100× reference crop and a 10× search image, find where the reference sits,
return its centre `(x, y)`, orientation `theta`, zoom `scale`, and a `found` flag
that says whether the reference is even present.

- **Phase 1 repo:** https://github.com/Jaswanthj006/Semicon-Submission

---

## Quick-start — how to test

### What you need

| File | Purpose |
|---|---|
| `register.py` | Entry point (the only script you run) |
| `localize.py` | Loads the model, runs inference |
| `train.py` | Pipeline core (proposals, verifier, refinement) |
| `model/verifier.pt` | Trained weights (ships in the ZIP) |
| `requirements.txt` | All dependencies |

---

## How to Test

### Step 1 — Get into the right folder

When you download the ZIP from GitHub it creates a nested folder.
You need to be inside the folder that contains `register.py`.

**Windows PowerShell:**
```powershell
cd "C:\Users\<you>\Downloads\Semicon-Submission-Phase2-main"
cd ".\Semicon-Submission-Phase2-main"
dir
```

You should see `register.py`, `localize.py`, `train.py`, `requirements.txt`, and a `model` folder.
If you do not see them, keep doing `cd <folder-name>` until you do.

---

### Step 2 — Create a virtual environment and install packages

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You will see packages downloading. Wait until the prompt returns.
You only need to do this once per machine.

---

### Step 3 — Run on the test set

> ⚠️ The full command must be typed **on one line**. Do not split it with `\` in PowerShell.

**Windows PowerShell:**
```powershell
python register.py --input "C:\Users\<you>\Downloads\phase2_test_set\pairs.csv" --output predictions.csv
```

**macOS / Linux:**
```bash
python register.py --input /path/to/phase2_test_set/pairs.csv --output predictions.csv
```

When it finishes you will see:
```
wrote 450 rows -> predictions.csv
```

`predictions.csv` is saved in whichever folder your terminal is currently in,
unless you give a full path like `--output C:\Users\<you>\Desktop\predictions.csv`.

---

### Step 4 — What the output looks like

Open `predictions.csv` in Excel or any text editor:

```text
pair_id,x,y,theta,scale,found,score
p001,423.1200,318.7500,0.8700,9.9500,1,0.923411
p002,0.0000,0.0000,0.0000,0.0000,0,0.041200
```

| Column | Meaning |
|---|---|
| `x, y` | Match centre in the search image, subpixel, top-left origin |
| `theta` | Rotation in degrees, CCW positive |
| `scale` | Down-scaling factor, nominally 8–12 |
| `found` | 1 = reference is present and located, 0 = absent |
| `score` | Confidence (higher is more certain) |

- `found=1` → reference was located; `x y theta scale` are the result
- `found=0` → reference is absent in that image; pose columns are all 0
- One row per pair, same order as `pairs.csv`

---

### Common Windows problems

| Symptom | Fix |
|---|---|
| `requirements.txt: No such file` | You are in the outer ZIP folder — `cd ".\Semicon-Submission-Phase2-main"` |
| `FileNotFoundError: phase2_test_set.pairs.csv` | The file is `phase2_test_set\pairs.csv` (a folder, then `pairs.csv`) |
| `source` not recognised | PowerShell uses `.\.venv\Scripts\Activate.ps1` |
| `Missing expression after --` | The full command must be on **one line**, no `\` breaks |
| `No module named 'cv2'` | Activate `.venv` first, then re-run `pip install -r requirements.txt` |
| numpy build fails / no compiler | Use Python **3.11** — pre-built wheels are available for it |

---

## What we improved from Phase 1

Phase 1 solved the basic location problem: ZNCC proposals → CNN verifier → sub-pixel `(x, y)`.
Phase 2 keeps that core and layers on top everything the new contract requires.

| Area | Phase 1 | Phase 2 |
|---|---|---|
| Entry point | `localize.py` prints `x y` | `register.py` → `predictions.csv` |
| Output columns | `x, y` | `x, y, theta, scale, found, score` |
| Scale search range | 9.0–11.0 (5 steps) | **8.0–12.0 (9 steps, Δ0.5)** |
| Rotation search range | ±2° (5 steps) | **±5° (11 steps, Δ1°)** |
| Pose grid total | 25 ZNCC maps | **99 ZNCC maps** |
| Absent pairs | Not scored | **Confidence gate → `found` flag** |
| Pose output | Not required | Discrete grid + **local pose refinement** |
| Speed | Full multi-pose ZNCC | **Coarse-to-fine pruning** (half-res first) |
| Weights path | Relative to CWD | **Script-relative** `model/verifier.pt` (safe from any CWD) |

**Core idea unchanged:** ZNCC generates candidate locations at multiple scales and rotations →
the CNN verifier picks the best cell → sub-pixel refinement sharpens `(x, y)` →
local pose refinement sharpens `theta` and `scale` → confidence gate decides `found`.

---

## Score on the test set

Evaluated on the held-out **`phase2_test_set`** (450 pairs — Set A: 180, Set B: 180, Set C: 90).
Run with the same `register.py` path.

### Rejection (Set C — absent pairs)

| Metric | Result |
|---|---|
| Absent pairs correctly rejected | 77 / 90 — **85.6%** |
| Present pairs wrongly rejected | 12 / 360 — 3.3% |
| F1 (`found`) | **0.965** |
| Precision / Recall | 0.964 / 0.967 |

### Localisation (present pairs, `found=1`)

| Set | Pairs | p@1 px | p@2 px | p@3 px | Median error |
|---|---:|---:|---:|---:|---:|
| All | 348 | 0.89 | 0.99 | 1.00 | **0.3 px** |
| Set A | 176 | 0.97 | 1.00 | 1.00 | 0.3 px |
| Set B | 172 | 0.81 | 0.98 | 0.99 | 0.4 px |

### Pose accuracy and runtime

| Metric | Result |
|---|---|
| Weighted loc credit (0.45×A + 0.55×B) | **0.939** |
| Median \|Δtheta\| | 0.09° |
| Median \|Δscale\| | 0.50% |
| Median time per pair (CPU) | **0.569 s** (budget: 5 s) |

---

## Failures we hit and how we fixed them

**1. Verifier weights not found when run from a different folder**  
The original code looked for `model/verifier.pt` relative to where you *run* the script.
If the organiser's scorer runs from a parent directory, the weights silently disappear
and every pair falls back to plain ZNCC (much worse accuracy).
Fix: `localize.py` now resolves the weights relative to its own file path, not the CWD.

**2. One bad pair crashed the whole run**  
If a single image was unreadable (bad path, truncated file), Python would raise
an exception and all remaining pairs would produce no output — a missing row scores zero,
so this could silently lose hundreds of pairs.
Fix: `register.py` wraps every pair in a `try/except`. A failed pair writes `found=0`
and processing continues.

**3. DRAM lattice ambiguity**  
DRAM dies have highly periodic patterns. ZNCC produces many equally-strong peaks
at the wrong lattice repeat positions. Phase 1 already used a CNN verifier with hard
negatives for this, but the narrow pose range (±2°, 9–11×) missed pairs where
the chip was tilted or zoomed further out.
Fix: Widened the search to ±5° and 8–12× with a coarse-to-fine strategy so runtime
stays under 1 s even with 99 pose maps.

**4. Runtime blowup with wider pose grid**  
Going from 25 to 99 full-resolution ZNCC maps pushed time per pair above the 5 s budget.
Fix: Run ZNCC at half resolution first, keep only the top-scoring `(scale, theta)` candidates,
then run full-resolution ZNCC only on those. Median time dropped from ~2.1 s to ~0.57 s.

---

## Method summary

```
pairs.csv
   │
   ▼
ZNCC at 9 (scale, theta) coarse candidates          ← half-res, fast
   │  prune to top-3 candidates
   ▼
ZNCC at full resolution for selected candidates
   │  top-K peak locations per map
   ▼
CNN Verifier (model/verifier.pt)                    ← ranks all peaks
   │  picks best (x, y, scale, theta)
   ▼
Sub-pixel refinement (parabolic fit)
   │
   ▼
Local pose refinement (gradient search around best peak)
   │
   ▼
Confidence gate  →  found=1 or found=0
   │
   ▼
predictions.csv
```

Weights ship in `model/verifier.pt` — nothing is downloaded at run time.
