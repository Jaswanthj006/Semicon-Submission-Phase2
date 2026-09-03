# DriftLoc — Semicon Phase 2 Submission

Locate a high-mag reference pattern inside a wider search image.  
Returns centre `(x, y)`, pose `(theta, scale)`, `found`, and confidence `score`.

Repo: https://github.com/Jaswanthj006/Semicon-Submission-Phase2

---

## Requirements

- **Python 3.11** (default evaluation environment)
- Dependencies in `requirements.txt`
- Weights at `model/verifier.pt`

---

## Repo layout

```text
register.py            # entry point
localize.py
train.py
generate_dataset.py
requirements.txt
model/verifier.pt
README.md
```

---

## Setup

### 1. Open the project folder

If you cloned the repo:

```powershell
cd Semicon-Submission-Phase2
```

If you downloaded the GitHub ZIP on Windows, go into the **inner** folder that contains `register.py` (ZIP downloads often nest twice):

```powershell
cd Semicon-Submission-Phase2-main\Semicon-Submission-Phase2-main
dir
```

You should see `register.py`, `requirements.txt`, and `model\`.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick check:

```powershell
python -c "import cv2, torch, numpy; print('ok')"
```

---

## Run (official interface)

`--input` must be a **`pairs.csv` file**, not an image folder.

**Windows PowerShell — use one line** (do not split with `\`):

```powershell
python register.py --input "C:\path\to\pairs.csv" --output predictions.csv
```

**macOS / Linux:**

```bash
python register.py --input /path/to/pairs.csv --output predictions.csv
```

### Input format (`pairs.csv`)

```text
pair_id,search_path,reference_path
p001,search/p001.png,reference/p001.png
```

Typical folder layout:

```text
test_set/
  pairs.csv
  reference/p001.png
  search/p001.png
```

Paths in the CSV are relative to the folder that contains `pairs.csv`.

### Output (`predictions.csv`)

```text
pair_id,x,y,theta,scale,found,score
```

- `(x, y)` = centre in the **search** image (pixels, top-left origin)
- `theta` = degrees, counter-clockwise positive
- `scale` = relative zoom
- `found=0` → `x/y/theta/scale` are written as `0`

---

## Windows notes (teammates)

| Problem | Fix |
|---|---|
| `requirements.txt` not found | You’re in the outer ZIP folder — `cd` one level deeper |
| `source` not recognized | Use `.\.venv\Scripts\Activate.ps1` |
| `Missing expression after '--'` | Put the full `register.py` command on **one line** |
| `--input` points at an image folder | Pass `pairs.csv` instead |
| `No module named 'cv2'` | Activate `.venv`, then finish `pip install -r requirements.txt` |
| pip tries to build numpy / compiler errors | Wrong Python (e.g. 3.14). Recreate the venv with the machine’s **3.11** interpreter |

Evaluation machines already have Python 3.11; no extra Python install is needed there.

---

## Method (short)

ZNCC multi-scale/rotation proposals → CNN verifier → sub-pixel + pose refine → confidence reject for absent cases.  
Weights ship in `model/verifier.pt` and are loaded relative to the script path.
