# Setup Guide

Target: **Python 3.12** and PettingZoo's `sisl/pursuit-v5` environment. PettingZoo does not
officially support native Windows; if installation fails there, use WSL2 (see the end).

## 1. Clone

```
git clone <REPO_URL>
cd Multi-Agent-RL-
```

## 2. Create an environment (pick ONE)

### Option A: venv (default)

macOS / Linux:

```
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.

### Option B: conda (alternative)

Use Anaconda Prompt (Windows) or your normal terminal (macOS/Linux):

```
conda create -n marl python=3.12 -y
conda activate marl
```

Install packages with pip (next step), not `conda install`, so there is one source of truth.

## 3. Install

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs PettingZoo with the SISL extras, pytest, and this project's own code
(`phase1_pursuit`, `phase2_unity_soccer`) in editable mode.

## 4. Verify

```
python -m phase1_pursuit.smoke_test
python -m pytest -q
```

Expected: `SUCCESS: Pursuit environment is working.` and all tests passing.
Post the smoke-test output in Discord. Add `--render` to watch a window.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named phase1_pursuit` | Environment not active, or step 3 not run. Re-run step 3. |
| `No module named pettingzoo` | Activate your environment, then re-run step 3. |
| `Environment version ... not found` | Use `sisl/pursuit-v5`; the version lives in `src/phase1_pursuit/env.py`. |
| `python` not recognized (Windows) | Use `py -3.12`. |
| `conda` not recognized | Use Anaconda/Miniforge Prompt, or run `conda init` and reopen the terminal. |
| Rendering fails | Test headless first; rendering needs platform graphics support. |

## Windows: WSL2 fallback

```
wsl --install
```

Restart, open Ubuntu, then:

```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Clone the repo inside the Linux filesystem and repeat the steps above.
