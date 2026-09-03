# DriftLoc — Semicon Phase 2

Find a 100× reference pattern inside a 10× search image.  
Phase 2 also returns pose (`theta`, `scale`) and a reject flag (`found`).

- Phase 2 repo: https://github.com/Jaswanthj006/Semicon-Submission-Phase2  
- Phase 1 repo: https://github.com/Jaswanthj006/Semicon-Submission  

---

## 1. Install and get output

### Requirements
- Python **3.11**
- Files: `register.py`, `localize.py`, `train.py`, `model/verifier.pt`, `requirements.txt`

### Setup (Windows PowerShell)

If you downloaded the GitHub ZIP, go into the **inner** folder that contains `register.py`:

```powershell
cd "C:\Users\<you>\Downloads\Semicon-Submission-Phase2-main"
cd ".\Semicon-Submission-Phase2-main"
dir
```

Then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
cd Semicon-Submission-Phase2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run (official command — one line)

`--input` must be a **`pairs.csv` file**, not an image folder.

```powershell
python register.py --input "C:\path\to\phase2_test_set\pairs.csv" --output predictions.csv
```

macOS / Linux:

```bash
python register.py --input /path/to/pairs.csv --output predictions.csv
```

### Input
`pairs.csv` next to `reference/` and `search/`:

```text
pair_id,search_path,reference_path
p001,search/p001.png,reference/p001.png
```

### Output
`predictions.csv` is written in the **current folder** (unless you give a full path):

```text
pair_id,x,y,theta,scale,found,score
```

- `(x, y)`: centre in the search image (pixels, top-left origin)
- `theta`: degrees, CCW positive
- `scale`: relative zoom
- `found=0` → `x/y/theta/scale` are `0`
- No progress bar in `register.py`; when finished you see `wrote N rows -> predictions.csv`

---

## 2. What we improved from Phase 1

Phase 1 ([Semicon-Submission](https://github.com/Jaswanthj006/Semicon-Submission)) solved basic localization: propose with ZNCC → CNN rank → sub-pixel `(x, y)`.

| Item | Phase 1 | Phase 2 (this repo) |
|---|---|---|
| Entry point | `localize.py` → prints `x y` | `register.py` → full `predictions.csv` |
| Output | `x, y` (+ confidence in CSV mode) | `x, y, theta, scale, found, score` |
| Scale search | 9.0–11.0 (5 values) | **8.0–12.0** (9 values, step 0.5) |
| Rotation search | ±2° (5 values) | **±5°** (11 values, step 1°) |
| Pose grid | 25 ZNCC maps | **99** ZNCC maps |
| Absent / reject | Not a Phase 1 scored output | **`found` gate** for Set C |
| Pose columns | Not required | Discrete grid + **local pose refine** |
| Speed | Full multi-pose ZNCC | **Coarse-to-fine** proposal prune |
| Weights path | Script / `model` | Script-relative `model/verifier.pt` (safe if cwd differs) |

**Same core idea kept:** ZNCC proposes look-alike DRAM peaks → CNN verifier picks the right cell → refine.  
**Phase 2 adds:** wider pose range, reject decision, pose reporting, and runtime optimizations for the Phase 2 contract.

---

## 3. Score on test data

Held-out **`phase2_test_set`** (450 pairs: Set A 180 / B 180 / C 90).  
Same inference path as `register.py`.

### Rejection

| Metric | Result |
|---|---|
| Absent correct reject | 77/90 (85.6%) |
| Present false reject | 12/360 (3.3%) |
| F1 (`found`) | **0.965** |
| Precision / Recall | 0.964 / 0.967 |

### Localization (present + `found=1`)

| Set | n | p@1 | p@2 | p@3 | median err |
|---|---:|---:|---:|---:|---:|
| ALL | 348 | 0.89 | 0.99 | 1.00 | 0.3 px |
| A | 176 | 0.97 | 1.00 | 1.00 | 0.3 px |
| B | 172 | 0.81 | 0.98 | 0.99 | 0.4 px |

### Pose & runtime (accepted present)

| Metric | Result |
|---|---|
| Approx. loc credit A / B / weighted (0.45A+0.55B) | 0.971 / 0.912 / **0.939** |
| Median \|Δθ\| | 0.09° |
| Median \|Δscale\| | 0.50% |
| Median time / pair | **0.569 s** |

---

## Method (short)

ZNCC multi-scale/rotation proposals → CNN verifier → sub-pixel refine → local pose refine → confidence reject.  
Weights: `model/verifier.pt`.

---

## Windows tips

| Problem | Fix |
|---|---|
| `requirements.txt` not found | `cd ".\Semicon-Submission-Phase2-main"` (inner ZIP folder) |
| `phase2_test_set.pairs.csv` not found | Use `phase2_test_set\pairs.csv` |
| `source` not recognized | Use `.\.venv\Scripts\Activate.ps1` |
| Command split with `\` fails | Put `register.py` on **one line** |
| `No module named 'cv2'` | Activate `.venv`, then finish `pip install -r requirements.txt` |
| pip tries to build numpy / compiler errors | Use Python **3.11** for the venv |

Evaluation machines already have Python 3.11; no extra Python install is needed there.
